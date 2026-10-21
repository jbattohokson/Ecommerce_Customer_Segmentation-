"""Write the portfolio charts (PNG) and summary tables (CSV) from the built warehouse.

Every figure is read from the dbt marts and the segmentation tables, so the charts
can't drift from the README or the app. Run after `dbt build` and `ml/segmentation.py`:

    .venv/bin/python reports/make_outputs.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "outputs"
DEFAULT_DB = Path(os.environ.get("RETAIL_DB_PATH", PROJECT_ROOT / "retail_analytics.duckdb"))

INK, MUTED, GRID = "#0f1117", "#4a4f5e", "#dde1ea"
# One fixed color per segment, so a segment keeps its color in every chart.
SEGMENT_COLORS = {
    "Loyal Champions": "#1a56db",
    "Recent Newcomers": "#1a7f4b",
    "At-Risk Whales": "#b45309",
    "Dormant Buyers": "#8b93a8",
}
SEGMENT_ORDER = list(SEGMENT_COLORS)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 160, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})
GBP = FuncFormatter(lambda v, _: f"£{v / 1e6:.1f}M" if abs(v) >= 1e6 else f"£{v / 1e3:.0f}k")


def q(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote outputs/{name}")


def segment_shares(con) -> pd.DataFrame:
    df = q(con, """SELECT segment_name, is_churn_risk, customers, median_recency_days, median_frequency,
                          median_monetary, mean_monetary, total_monetary, customer_share, revenue_share
                   FROM marts.ml_segment_profiles""")
    df = df.set_index("segment_name").loc[SEGMENT_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["customer_share"], width=0.38, color="#c9cfdb", label="Share of customers")
    ax.bar([i + 0.2 for i in x], df["revenue_share"], width=0.38,
           color=[SEGMENT_COLORS[s] for s in df["segment_name"]], label="Share of net revenue")
    for i, (c, r) in enumerate(zip(df["customer_share"], df["revenue_share"])):
        ax.text(i - 0.2, c + 0.012, f"{c:.0%}", ha="center", fontsize=9, color=MUTED)
        ax.text(i + 0.2, r + 0.012, f"{r:.1%}", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(list(x), df["segment_name"])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    ax.set_title("20% of customers bring in 74% of net revenue")
    save(fig, "chart1_customer_vs_revenue_share.png")
    return df


def rfm_scatter(con) -> None:
    df = q(con, "SELECT segment_name, recency_days, frequency, monetary FROM marts.dim_customer_segments")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for seg in SEGMENT_ORDER:
        part = df[df["segment_name"] == seg]
        ax.scatter(part["recency_days"], part["monetary"], s=6, alpha=0.45, color=SEGMENT_COLORS[seg],
                   label=f"{seg} ({len(part):,})", linewidths=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.set_xlabel("Days since last order (recency)")
    ax.set_ylabel("Net lifetime spend (log scale)")
    ax.legend(frameon=False, markerscale=3, loc="lower left")
    ax.set_title("K-Means segments in recency and spend (k = 4, log-scaled RFM)")
    save(fig, "chart2_rfm_segments.png")


def monthly(con) -> pd.DataFrame:
    df = q(con, "SELECT * FROM marts.agg_monthly_sales ORDER BY month_start")
    df["month_start"] = pd.to_datetime(df["month_start"])
    full = df[~df["is_partial_month"]]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(full["month_start"], full["net_revenue"], color="#1a56db", marker="o", markersize=3.5, linewidth=1.6)
    ax.yaxis.set_major_formatter(GBP)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Net revenue")
    ax.set_title("Monthly net revenue peaks in Oct-Nov both years (partial months excluded)")
    save(fig, "chart3_monthly_net_revenue.png")
    return df


def k_selection(con) -> pd.DataFrame:
    df = q(con, "SELECT * FROM marts.ml_k_selection ORDER BY k")
    from ml.segmentation import CANDIDATE_K

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axvspan(min(CANDIDATE_K) - 0.4, max(CANDIDATE_K) + 0.4, color="#e8eefb", zorder=0,
               label=f"candidate range k = {min(CANDIDATE_K)}-{max(CANDIDATE_K)} (fewer than 4 can't map to four campaigns)")
    ax.plot(df["k"], df["silhouette"], color=INK, marker="o")
    chosen = df[df["is_chosen"]]
    ax.scatter(chosen["k"], chosen["silhouette"], s=120, color="#1a56db", zorder=3,
               label=f"chosen k = {int(chosen['k'].iloc[0])} (silhouette {chosen['silhouette'].iloc[0]:.3f})")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_ylim(top=df["silhouette"].max() + 0.06)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("Choosing k: best silhouette inside the candidate range")
    save(fig, "chart4_k_selection.png")
    return df


def products(con) -> pd.DataFrame:
    # Net of cancellations: the raw data has single orders of 80,995 and 74,215 units that were
    # cancelled in full, which would otherwise rank among the top products.
    df = q(con, """
        WITH returns AS (
            SELECT stock_code, -sum(line_total) AS returned_value, -sum(quantity) AS units_returned
            FROM main.raw_returns GROUP BY stock_code)
        SELECT p.stock_code, p.description, p.units_sold, coalesce(r.units_returned, 0) AS units_returned,
               p.gross_revenue, coalesce(r.returned_value, 0) AS returned_value,
               p.gross_revenue - coalesce(r.returned_value, 0) AS net_revenue, p.n_customers, p.median_unit_price
        FROM marts.dim_products p LEFT JOIN returns r USING (stock_code)
        ORDER BY net_revenue DESC LIMIT 25""")
    top = df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(top["description"].str.title().str.slice(0, 38), top["net_revenue"], color="#1a56db", height=0.62)
    ax.xaxis.set_major_formatter(GBP)
    ax.grid(axis="y", visible=False)
    ax.set_title("Top 10 products by net revenue (after returns and cancellations)")
    save(fig, "chart5_top_products.png")
    return df


def countries(con) -> pd.DataFrame:
    df = q(con, """SELECT c.country, count(*) AS customers, sum(t.monetary) AS net_revenue
                   FROM marts.fct_customer_transactions t JOIN marts.dim_customers c USING (customer_id)
                   GROUP BY 1 ORDER BY net_revenue DESC""")
    df["revenue_share"] = df["net_revenue"] / df["net_revenue"].sum()
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    con = duckdb.connect(args.db, read_only=True)
    try:
        tables = {"summary_by_segment.csv": segment_shares(con)}
        rfm_scatter(con)
        tables["summary_by_month.csv"] = monthly(con)
        tables["k_selection.csv"] = k_selection(con)
        tables["top_25_products.csv"] = products(con)
        tables["revenue_by_country.csv"] = countries(con)
    finally:
        con.close()
    for name, df in tables.items():
        df.round({c: 4 for c in df.select_dtypes("number").columns}).to_csv(OUT / name, index=False)
        print(f"  wrote outputs/{name} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
