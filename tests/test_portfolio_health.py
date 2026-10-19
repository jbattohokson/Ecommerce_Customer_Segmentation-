"""Pipeline health checks against the real, fully built retail_analytics.duckdb.

These run after fetch -> load -> dbt build -> segmentation (locally or in the weekly GitHub
Actions job). Unlike the unit tests, they check the actual warehouse: that it exists and is
populated, that keys are clean, that every active customer has exactly one segment, and that
the numbers still reconcile across layers.

If the database hasn't been built, every test here fails (not skips): in CI a missing database
means the pipeline broke, and that should trigger the alert.
"""

import math

import duckdb
import pytest

from ml.segmentation import CHURN_RISK_PERSONAS, SEGMENT_ORDER, load_model_from_db
from pipelines.load_to_duckdb import DB_PATH

# Sanity floors for the real UCI Online Retail II data (actual: ~777k lines, 5,852 customers).
# Generous on purpose: they catch a broken or truncated load, not normal variation.
MIN_SALE_LINES = 700_000
MIN_CUSTOMERS = 5_000


@pytest.fixture(scope="module")
def conn():
    assert DB_PATH.exists(), f"Database file {DB_PATH} is missing; the pipeline did not run."
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


def scalar(conn, sql):
    return conn.execute(sql).fetchone()[0]


# ---- Database exists and holds valid rows ---------------------------------------------------

def test_database_exists():
    assert DB_PATH.exists(), f"Database file {DB_PATH} missing!"


@pytest.mark.parametrize("table", [
    "main.raw_transactions", "main.raw_returns", "main.audit_rejected_rows",
    "marts.fct_invoice_lines", "marts.fct_customer_transactions", "marts.dim_customers",
    "marts.dim_products", "marts.dim_date", "marts.agg_monthly_sales",
    "marts.dim_customer_segments", "marts.ml_segment_profiles",
])
def test_table_exists_and_is_not_empty(conn, table):
    assert scalar(conn, f"SELECT count(*) FROM {table}") > 0, f"{table} is empty"


def test_raw_load_is_complete(conn):
    assert scalar(conn, "SELECT count(*) FROM main.raw_transactions") >= MIN_SALE_LINES
    assert scalar(conn, "SELECT count(DISTINCT customer_id) FROM main.raw_transactions") >= MIN_CUSTOMERS


def test_load_accounted_for_every_raw_row(conn):
    summary = scalar(conn, "SELECT sum(row_count) FROM main.audit_load_summary")
    tables = scalar(conn, """SELECT (SELECT count(*) FROM main.raw_transactions)
                                  + (SELECT count(*) FROM main.raw_returns)
                                  + (SELECT count(*) FROM main.audit_rejected_rows)""")
    assert summary == tables


def test_sales_values_are_valid(conn):
    bad = scalar(conn, """SELECT count(*) FROM main.raw_transactions
                          WHERE quantity <= 0 OR unit_price <= 0 OR invoice_ts IS NULL
                             OR upper(invoice_no) LIKE 'C%'""")
    assert bad == 0


# ---- No null customer IDs in the marts --------------------------------------------------------

@pytest.mark.parametrize("table", [
    "marts.dim_customer_segments", "marts.fct_customer_transactions", "marts.fct_invoice_lines",
    "marts.dim_customers",
])
def test_no_null_customer_ids(conn, table):
    nulls = scalar(conn, f"SELECT count(*) FROM {table} WHERE customer_id IS NULL")
    assert nulls == 0, f"Data integrity error: {nulls} null customer IDs in {table}"


def test_segment_customer_ids_are_unique(conn):
    assert scalar(conn, """SELECT count(*) - count(DISTINCT customer_id)
                           FROM marts.dim_customer_segments""") == 0


# ---- K-Means assigns every active customer, with no NaN groups -------------------------------

def test_every_active_customer_is_segmented(conn):
    active = scalar(conn, "SELECT count(*) FROM marts.fct_customer_transactions WHERE monetary > 0")
    unassigned = scalar(conn, """
        SELECT count(*) FROM marts.fct_customer_transactions t
        LEFT JOIN marts.dim_customer_segments s USING (customer_id)
        WHERE t.monetary > 0 AND s.customer_id IS NULL""")
    assert unassigned == 0, f"{unassigned} of {active} active customers have no segment"
    assert scalar(conn, "SELECT count(*) FROM marts.dim_customer_segments") == active


def test_no_unassigned_or_nan_segments(conn):
    bad = scalar(conn, """
        SELECT count(*) FROM marts.dim_customer_segments
        WHERE cluster_id IS NULL OR segment_name IS NULL OR trim(segment_name) = ''
           OR isnan(recency_days) OR isnan(frequency) OR isnan(monetary)""")
    assert bad == 0


def test_only_known_segments_and_all_four_present(conn):
    names = {r[0] for r in conn.execute(
        "SELECT DISTINCT segment_name FROM marts.dim_customer_segments").fetchall()}
    assert names == set(SEGMENT_ORDER), f"Unexpected segment set: {sorted(names)}"


def test_kmeans_produced_meaningful_clusters(conn):
    sizes = [r[0] for r in conn.execute(
        "SELECT customers FROM marts.ml_segment_profiles ORDER BY customers").fetchall()]
    total = sum(sizes)
    assert len(sizes) > 1, "ML pipeline failure: K-Means produced a single segment"
    assert min(sizes) / total >= 0.05, f"A segment holds under 5% of customers: {sizes}"


def test_churn_flag_matches_segment(conn):
    mismatched = scalar(conn, f"""
        SELECT count(*) FROM marts.dim_customer_segments
        WHERE is_churn_risk <> (segment_name IN {tuple(sorted(CHURN_RISK_PERSONAS))})""")
    assert mismatched == 0


def test_segments_follow_their_persona(conn):
    """Champions should be more recent and spend more than Dormant Buyers; otherwise the
    persona naming has drifted from what the clusters actually contain."""
    p = {r[0]: r[1:] for r in conn.execute(
        """SELECT segment_name, median_recency_days, median_monetary
           FROM marts.ml_segment_profiles""").fetchall()}
    champ_r, champ_m = p["Loyal Champions"]
    dorm_r, dorm_m = p["Dormant Buyers"]
    whale_r, _ = p["At-Risk Whales"]
    assert champ_r < whale_r and champ_r < dorm_r
    assert champ_m > dorm_m


def test_saved_model_reproduces_assignments(conn):
    model = load_model_from_db(conn)
    seg = conn.execute("SELECT * FROM marts.dim_customer_segments ORDER BY customer_id").df()
    agree = (model.predict(seg) == seg["cluster_id"].to_numpy()).mean()
    assert agree == 1.0, f"Stored model params reproduce only {agree:.2%} of assignments"


# ---- Cross-layer reconciliation ----------------------------------------------------------------

def test_segment_value_reconciles_to_rfm_mart(conn):
    seg = scalar(conn, "SELECT sum(monetary) FROM marts.dim_customer_segments")
    rfm = scalar(conn, "SELECT sum(monetary) FROM marts.fct_customer_transactions WHERE monetary > 0")
    assert math.isclose(float(seg), float(rfm), rel_tol=1e-9)


def test_segment_profiles_sum_to_one(conn):
    shares = conn.execute(
        "SELECT sum(customer_share), sum(revenue_share) FROM marts.ml_segment_profiles").fetchone()
    assert all(math.isclose(s, 1.0, abs_tol=1e-3) for s in shares)


def test_segmentation_is_newer_than_the_load(conn):
    loaded, segmented = conn.execute("""
        SELECT (SELECT max(loaded_at) FROM main.audit_load_summary)::TIMESTAMP,
               (SELECT max(segmented_at) FROM marts.dim_customer_segments)""").fetchone()
    assert segmented >= loaded.replace(tzinfo=None), "Segments are stale: older than the last data load"
