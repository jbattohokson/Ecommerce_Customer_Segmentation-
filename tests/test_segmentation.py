"""Unit tests for ml/segmentation.py on synthetic RFM data with known personas."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from ml.segmentation import (
    CHURN_RISK_PERSONAS, SegmentationError, fit_segment_model, load_model_from_db,
    name_clusters, run_rfm_clustering,
)

# (recency_days, frequency, monetary) medians for four well-separated synthetic personas.
TRUTH = {
    "Loyal Champions":  (10, 20, 8000),
    "At-Risk Whales":   (300, 15, 6000),
    "Recent Newcomers": (12, 1, 150),
    "Dormant Buyers":   (400, 1, 120),
}


def synthetic_rfm(n_per=150, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i, (persona, (r, f, m)) in enumerate(TRUTH.items()):
        frames.append(pd.DataFrame({
            "customer_id": np.arange(n_per) + i * 10_000,
            "recency_days": np.clip(rng.lognormal(np.log(r), 0.15, n_per), 1, None).round(),
            "frequency": np.clip(rng.lognormal(np.log(f), 0.15, n_per), 1, None).round(),
            "monetary": rng.lognormal(np.log(m), 0.15, n_per),
            "truth": persona,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def fitted():
    rfm = synthetic_rfm()
    model, labels, diagnostics = fit_segment_model(rfm, k=4)
    return rfm, model, labels, diagnostics


def test_personas_are_recovered(fitted):
    rfm, model, labels, _ = fitted
    predicted = np.array(model.segment_names, dtype=object)[labels]
    assert (predicted == rfm["truth"].to_numpy()).mean() > 0.98


def test_names_are_unique_and_ordered(fitted):
    _, model, _, _ = fitted
    assert model.segment_names == ["Loyal Champions", "At-Risk Whales", "Recent Newcomers", "Dormant Buyers"]
    assert len(set(model.segment_names)) == 4


def test_every_customer_gets_a_cluster(fitted):
    rfm, _, labels, _ = fitted
    assert len(labels) == len(rfm)
    assert not np.isnan(labels.astype(float)).any()
    assert set(labels) == {0, 1, 2, 3}


def test_model_predict_matches_fit_labels(fitted):
    rfm, model, labels, _ = fitted
    np.testing.assert_array_equal(model.predict(rfm), labels)


def test_fit_is_deterministic():
    rfm = synthetic_rfm()
    _, a, _ = fit_segment_model(rfm, k=4)
    _, b, _ = fit_segment_model(rfm, k=4)
    np.testing.assert_array_equal(a, b)


def test_diagnostics_cover_range_and_mark_one_choice(fitted):
    *_, diagnostics = fitted
    assert list(diagnostics["k"]) == list(range(2, 9))
    assert diagnostics["is_chosen"].sum() == 1
    assert diagnostics.loc[diagnostics.is_chosen, "k"].item() == 4


def test_extra_clusters_get_descriptive_names():
    # Six centroids: the four persona corners plus two extras. Columns: recency, frequency, monetary.
    cents = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, -1], [1, -1, -1], [0.2, 0.3, 0.3], [-0.2, -0.3, -0.3]])
    names = name_clusters(cents)
    assert names[:4] == ["Loyal Champions", "At-Risk Whales", "Recent Newcomers", "Dormant Buyers"]
    assert names[4] == "Lapsing Higher-Value Group 4"
    assert names[5] == "Active Lower-Value Group 5"


def test_aging_customers_moves_them_toward_churn_risk(fitted):
    rfm, model, _, _ = fitted
    champs = rfm[rfm.truth == "Loyal Champions"].copy()
    champs["recency_days"] += 365
    names = set(model.predict_names(champs))
    assert names <= CHURN_RISK_PERSONAS


# ---- end-to-end against a temp DuckDB -------------------------------------------------------

@pytest.fixture()
def rfm_db(tmp_path):
    rfm = synthetic_rfm()
    rfm.loc[0, "monetary"] = -50.0  # a net returner: must be excluded, not clustered
    db = tmp_path / "seg.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE SCHEMA marts")
        conn.register("rfm", rfm.drop(columns="truth"))
        conn.execute("CREATE TABLE marts.fct_customer_transactions AS SELECT * FROM rfm")
    return db, rfm


def test_run_writes_all_tables_and_excludes_net_returners(rfm_db):
    db, rfm = rfm_db
    profiles = run_rfm_clustering(db, k=4)
    with duckdb.connect(str(db), read_only=True) as conn:
        seg = conn.execute("SELECT * FROM marts.dim_customer_segments").df()
        tables = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='marts'").fetchall()}
    assert {"dim_customer_segments", "ml_segment_profiles", "ml_k_selection", "ml_model_params"} <= tables
    assert len(seg) == len(rfm) - 1
    assert rfm.loc[0, "customer_id"] not in set(seg.customer_id)
    assert seg["segment_name"].notna().all() and seg["customer_id"].is_unique
    assert profiles["customers"].sum() == len(seg)
    assert profiles["revenue_share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_model_round_trips_through_duckdb(rfm_db):
    db, rfm = rfm_db
    run_rfm_clustering(db, k=4)
    with duckdb.connect(str(db), read_only=True) as conn:
        model = load_model_from_db(conn)
        seg = conn.execute("SELECT * FROM marts.dim_customer_segments ORDER BY customer_id").df()
    np.testing.assert_array_equal(model.predict(seg), seg["cluster_id"].to_numpy())


def test_missing_mart_raises_helpful_error(tmp_path):
    with pytest.raises(SegmentationError, match="dbt build"):
        run_rfm_clustering(tmp_path / "empty.duckdb")
