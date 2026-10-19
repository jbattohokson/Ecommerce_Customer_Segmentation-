"""K-Means RFM segmentation: marts.fct_customer_transactions -> marts.dim_customer_segments.

Pipeline:
  1. Pull recency_days / frequency / monetary per customer from the dbt mart. Customers whose net
     spend is <= 0 (they returned everything) are not "active" buyers and are left out.
  2. log1p-transform, then StandardScaler. RFM is extremely right-skewed here (median net spend
     ~£840, max ~£578k). On raw values K-Means "wins" on silhouette by isolating a handful of
     wholesale accounts in their own clusters, which is useless for marketing. log1p first
     gives balanced, interpretable clusters.
  3. Pick k from CANDIDATE_K by silhouette score (k < 4 is too coarse for four campaign
     personas). Diagnostics for every k in DIAGNOSTIC_K are saved for the dashboard.
  4. Name clusters by matching each centroid to a persona prototype in (recency, value) space
     with the Hungarian algorithm, so names are unique and don't depend on K-Means' arbitrary ids.
  5. Write, in one transaction, to the `marts` schema:
       dim_customer_segments   one row per active customer: cluster, segment, RFM, churn flag
       ml_segment_profiles     one row per segment: size, medians, revenue share
       ml_k_selection          silhouette / inertia / Davies-Bouldin per k, with the chosen k
       ml_model_params         scaler mean/scale per feature + centroids, so the app can
                               re-score "what-if" RFM values without a pickled model

Usage:
    python ml/segmentation.py
    python ml/segmentation.py --k 5          # force k instead of choosing by silhouette
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("segmentation")

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("RETAIL_DB_PATH", REPO_ROOT / "retail_analytics.duckdb"))

FEATURES = ["recency_days", "frequency", "monetary"]
RANDOM_STATE = 42
N_INIT = 10
CANDIDATE_K = (4, 5, 6)
DIAGNOSTIC_K = range(2, 9)
SILHOUETTE_SAMPLE = 4000  # silhouette is O(n^2); a fixed sample keeps it fast and deterministic

# Persona prototypes in (recency z, value z) space, where value z = mean of frequency z and
# monetary z. High recency = long since last purchase.
PERSONAS = {
    "Loyal Champions":  (-1.0, +1.0),
    "At-Risk Whales":   (+1.0, +1.0),
    "Recent Newcomers": (-1.0, -1.0),
    "Dormant Buyers":   (+1.0, -1.0),
}
# Personas whose customers are lapsing: the retention agent and app target these.
CHURN_RISK_PERSONAS = {"At-Risk Whales", "Dormant Buyers"}
# Stable display order: best customers first.
SEGMENT_ORDER = ["Loyal Champions", "At-Risk Whales", "Recent Newcomers", "Dormant Buyers"]


@dataclass
class SegmentModel:
    """Everything needed to score new RFM rows: scaler parameters and named centroids."""
    scaler_mean: np.ndarray       # per feature, in log1p space
    scaler_scale: np.ndarray
    centroids: np.ndarray         # (k, 3) in scaled log1p space, row i = cluster_id i
    segment_names: list[str]      # index = cluster_id

    def transform(self, rfm: pd.DataFrame) -> np.ndarray:
        x = np.log1p(rfm[FEATURES].to_numpy(dtype=float))
        return (x - self.scaler_mean) / self.scaler_scale

    def predict(self, rfm: pd.DataFrame) -> np.ndarray:
        """Nearest-centroid cluster ids (identical to KMeans.predict on the fitted model)."""
        z = self.transform(rfm)
        dists = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return dists.argmin(axis=1)

    def predict_names(self, rfm: pd.DataFrame) -> np.ndarray:
        return np.array(self.segment_names, dtype=object)[self.predict(rfm)]


class SegmentationError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# Data in / out
# --------------------------------------------------------------------------------------------

def load_rfm(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    try:
        df = conn.execute(
            "SELECT customer_id, recency_days, frequency, CAST(monetary AS DOUBLE) AS monetary "
            "FROM marts.fct_customer_transactions WHERE monetary > 0 ORDER BY customer_id"
        ).df()
    except duckdb.CatalogException as exc:
        raise SegmentationError(
            "marts.fct_customer_transactions not found. Run `DBT_PROFILES_DIR=. dbt build` first."
        ) from exc
    if len(df) < max(CANDIDATE_K) * 10:
        raise SegmentationError(f"Only {len(df)} active customers; too few to cluster.")
    if df[FEATURES].isna().any().any():
        raise SegmentationError("Null RFM values in marts.fct_customer_transactions.")
    return df


# --------------------------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------------------------

def scale_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    z = scaler.fit_transform(np.log1p(rfm[FEATURES].to_numpy(dtype=float)))
    return z, scaler


def fit_kmeans(z: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(z)


def evaluate_k(z: np.ndarray, ks=DIAGNOSTIC_K) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = fit_kmeans(z, k)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(z, km.labels_, sample_size=min(SILHOUETTE_SAMPLE, len(z)),
                                           random_state=RANDOM_STATE),
            "inertia": km.inertia_,
            "davies_bouldin": davies_bouldin_score(z, km.labels_),
            "min_cluster_size": int(np.bincount(km.labels_).min()),
        })
    return pd.DataFrame(rows)


def choose_k(diagnostics: pd.DataFrame, candidates=CANDIDATE_K) -> int:
    pool = diagnostics[diagnostics["k"].isin(candidates)]
    return int(pool.loc[pool["silhouette"].idxmax(), "k"])


def name_clusters(centroids: np.ndarray) -> list[str]:
    """Map each centroid to a unique persona name. Index of the result = cluster id.

    Centroids are in scaled (recency, frequency, monetary) space. Each is projected to
    (recency, value) and matched to PERSONAS by minimum total distance. With k > 4 the extra
    clusters get a descriptive name from their quadrant instead.
    """
    points = np.column_stack([centroids[:, 0], centroids[:, 1:].mean(axis=1)])
    protos = np.array(list(PERSONAS.values()))
    cost = ((points[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    names: list[str | None] = [None] * len(centroids)
    persona_names = list(PERSONAS)
    for r, c in zip(rows, cols):
        names[r] = persona_names[c]
    for i, name in enumerate(names):
        if name is None:
            recency, value = points[i]
            names[i] = (f"{'Lapsing' if recency > 0 else 'Active'} "
                        f"{'Higher' if value > 0 else 'Lower'}-Value Group {i}")
    return names  # type: ignore[return-value]


def relabel_by_display_order(km: KMeans, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Renumber clusters so cluster_id follows SEGMENT_ORDER (stable across reruns and k)."""
    def sort_key(i):
        return (SEGMENT_ORDER.index(names[i]) if names[i] in SEGMENT_ORDER else len(SEGMENT_ORDER), names[i])
    order = sorted(range(len(names)), key=sort_key)
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    return remap[km.labels_], km.cluster_centers_[order], [names[i] for i in order]


def fit_segment_model(rfm: pd.DataFrame, k: int | None = None) -> tuple[SegmentModel, np.ndarray, pd.DataFrame]:
    """Fit the full model. Returns (model, cluster id per row of rfm, k diagnostics)."""
    z, scaler = scale_features(rfm)
    diagnostics = evaluate_k(z)
    chosen = k or choose_k(diagnostics)
    diagnostics["is_chosen"] = diagnostics["k"] == chosen
    km = fit_kmeans(z, chosen)
    labels, centroids, names = relabel_by_display_order(km, name_clusters(km.cluster_centers_))
    model = SegmentModel(scaler.mean_, scaler.scale_, centroids, names)
    return model, labels, diagnostics


# --------------------------------------------------------------------------------------------
# Output tables
# --------------------------------------------------------------------------------------------

def build_segments(rfm: pd.DataFrame, model: SegmentModel, labels: np.ndarray) -> pd.DataFrame:
    out = rfm[["customer_id", *FEATURES]].copy()
    out["cluster_id"] = labels.astype(int)
    out["segment_name"] = np.array(model.segment_names, dtype=object)[labels]
    out["is_churn_risk"] = out["segment_name"].isin(CHURN_RISK_PERSONAS)
    # Distance to own centroid: lets the app show "borderline" customers near a boundary.
    z = model.transform(out)
    out["distance_to_centroid"] = np.linalg.norm(z - model.centroids[labels], axis=1).round(4)
    out["segmented_at"] = pd.Timestamp.now()  # local time, matching the loader's loaded_at
    return out[["customer_id", "cluster_id", "segment_name", "is_churn_risk", *FEATURES,
                "distance_to_centroid", "segmented_at"]]


def build_profiles(segments: pd.DataFrame) -> pd.DataFrame:
    total_customers = len(segments)
    total_value = segments["monetary"].sum()
    prof = (segments.groupby(["cluster_id", "segment_name", "is_churn_risk"])
            .agg(customers=("customer_id", "size"),
                 median_recency_days=("recency_days", "median"),
                 median_frequency=("frequency", "median"),
                 median_monetary=("monetary", "median"),
                 mean_monetary=("monetary", "mean"),
                 total_monetary=("monetary", "sum"))
            .reset_index())
    prof["customer_share"] = (prof["customers"] / total_customers).round(4)
    prof["revenue_share"] = (prof["total_monetary"] / total_value).round(4)
    return prof.sort_values("cluster_id").reset_index(drop=True)


def build_model_params(model: SegmentModel) -> pd.DataFrame:
    rows = [{"param_type": "scaler", "cluster_id": None, "segment_name": None, "feature": f,
             "value_1": float(model.scaler_mean[i]), "value_2": float(model.scaler_scale[i])}
            for i, f in enumerate(FEATURES)]
    for cid, (centroid, name) in enumerate(zip(model.centroids, model.segment_names)):
        rows += [{"param_type": "centroid", "cluster_id": cid, "segment_name": name, "feature": f,
                  "value_1": float(centroid[i]), "value_2": None} for i, f in enumerate(FEATURES)]
    return pd.DataFrame(rows)


def load_model_from_db(conn: duckdb.DuckDBPyConnection) -> SegmentModel:
    """Rebuild the SegmentModel from marts.ml_model_params (used by the app's simulator)."""
    p = conn.execute("SELECT * FROM marts.ml_model_params").df()
    scaler = p[p.param_type == "scaler"].set_index("feature").loc[FEATURES]
    cents = p[p.param_type == "centroid"]
    k = int(cents["cluster_id"].max()) + 1
    centroids = np.zeros((k, len(FEATURES)))
    names = [""] * k
    for _, r in cents.iterrows():
        centroids[int(r.cluster_id), FEATURES.index(r.feature)] = r.value_1
        names[int(r.cluster_id)] = r.segment_name
    return SegmentModel(scaler["value_1"].to_numpy(), scaler["value_2"].to_numpy(), centroids, names)


def write_tables(conn: duckdb.DuckDBPyConnection, tables: dict[str, pd.DataFrame]) -> None:
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS marts")
        for name, df in tables.items():
            conn.register("_df", df)
            conn.execute(f"CREATE OR REPLACE TABLE marts.{name} AS SELECT * FROM _df")
            conn.unregister("_df")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def run_rfm_clustering(db_path: Path = DB_PATH, k: int | None = None) -> pd.DataFrame:
    """Fit, label and write the segmentation. Returns the segment profile table."""
    conn = duckdb.connect(str(db_path))
    try:
        rfm = load_rfm(conn)
        model, labels, diagnostics = fit_segment_model(rfm, k)
        segments = build_segments(rfm, model, labels)
        if segments["segment_name"].isna().any() or len(segments) != len(rfm):
            raise SegmentationError("Some active customers were left without a segment.")
        profiles = build_profiles(segments)
        write_tables(conn, {
            "dim_customer_segments": segments,
            "ml_segment_profiles": profiles,
            "ml_k_selection": diagnostics,
            "ml_model_params": build_model_params(model),
        })
    finally:
        conn.close()

    chosen = diagnostics.loc[diagnostics.is_chosen].iloc[0]
    log.info("Segmented %s active customers into k=%d clusters (silhouette %.3f)",
             f"{len(segments):,}", chosen.k, chosen.silhouette)
    for _, r in profiles.iterrows():
        log.info("  %-18s %5d customers  recency %4.0fd  freq %3.0f  monetary £%9s  revenue %5.1f%%",
                 r.segment_name, r.customers, r.median_recency_days, r.median_frequency,
                 f"{r.median_monetary:,.0f}", 100 * r.revenue_share)
    return profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--k", type=int, default=None, help="force the number of clusters")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        run_rfm_clustering(args.db_path, args.k)
    except SegmentationError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
