"""Load and clean the UCI Online Retail II CSVs into retail_analytics.duckdb.

Every raw row lands in exactly one place, so nothing disappears silently:

    raw_transactions     clean sale lines: product, known customer, quantity > 0, price > 0
    raw_returns          cancellations ('C' invoices) of product lines by a known customer
    audit_rejected_rows  everything else, with a reject_reason (first matching rule wins)
    audit_load_summary   one row per destination/reason with its row count, for the dashboard + tests

Reject rules, in the order they're applied:
    unparseable_value     Quantity / Price / InvoiceDate / Customer ID that won't cast
    cross_file_overlap    the two UCI sheets both contain 2010-12-01..09; an invoice already seen in
                          an earlier file is dropped from later ones
    exact_duplicate       a line identical in every column to an earlier line (repeat scans)
    bad_debt_adjustment   'A' invoices (accounting write-offs, not sales)
    non_product_code      postage, carriage, fees, manual adjustments, samples, test items, vouchers
    missing_customer_id   can't be attributed to a customer, so useless for RFM
    non_positive_price    price <= 0 (free samples, data-entry fixes)
    non_cancel_negative   negative quantity without a 'C' invoice (stock write-offs / damages)

The whole load runs in one transaction and is checked before commit (row conservation, key
columns not null, value ranges); any failure rolls back and leaves the previous load intact.

Usage:
    python pipelines/load_to_duckdb.py
    python pipelines/load_to_duckdb.py --raw-dir data/raw --db-path retail_analytics.duckdb
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import duckdb

log = logging.getLogger("load_to_duckdb")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
DB_PATH = Path(os.environ.get("RETAIL_DB_PATH", REPO_ROOT / "retail_analytics.duckdb"))

# Header name -> snake_case column. Accepts both the Online Retail II names and the older
# Online Retail I names (InvoiceNo / UnitPrice / CustomerID) that the project spec uses.
COLUMN_ALIASES = {
    "invoice": "invoice_no", "invoiceno": "invoice_no",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "unit_price", "unitprice": "unit_price",
    "customer id": "customer_id", "customerid": "customer_id",
    "country": "country",
}
REQUIRED_COLUMNS = ["invoice_no", "stock_code", "description", "quantity",
                    "invoice_date", "unit_price", "customer_id", "country"]

# Stock codes that are charges, fees or adjustments rather than products. Real products are
# 5-digit codes with an optional letter suffix (85123A), plus a few DCGS*/SP* catalogue items.
NON_PRODUCT_CODE_REGEX = (
    r"^(POST|DOT|M|C2|C3|D|S|B|CRUK|PADS|GIFT|BANK CHARGES|AMAZONFEE"
    r"|ADJUST[0-9]*|TEST[0-9]*|gift_.*)$"
)

DESTINATION_TABLES = ["raw_transactions", "raw_returns", "audit_rejected_rows", "audit_load_summary"]


class LoadError(RuntimeError):
    """Raised when the raw files or the loaded result fail a contract or integrity check."""


def normalise_header(name: str) -> str:
    return COLUMN_ALIASES.get(name.strip().lower(), name.strip().lower())


def discover_csvs(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs in {raw_dir}. Run `python pipelines/fetch_data.py` to download the "
            "UCI Online Retail II dataset first.")
    return files


def read_header(conn: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    cols = conn.execute("DESCRIBE SELECT * FROM read_csv(?, all_varchar=true, header=true)",
                        [str(path)]).fetchall()
    return [c[0] for c in cols]


def stage_raw_files(conn: duckdb.DuckDBPyConnection, files: list[Path]) -> int:
    """Union all CSVs into temp table _raw as text, with file order and line number kept."""
    selects = []
    for file_idx, path in enumerate(files):
        header = read_header(conn, path)
        mapped = {normalise_header(h): h for h in header}
        missing = [c for c in REQUIRED_COLUMNS if c not in mapped]
        if missing:
            raise LoadError(f"{path.name} is missing required columns {missing} (header: {header})")
        cols = ", ".join(f'"{mapped[c]}" AS {c}' for c in REQUIRED_COLUMNS)
        # Path goes in as a SQL literal because DuckDB can't bind parameters inside a UNION view.
        literal = str(path).replace("'", "''")
        selects.append(
            f"SELECT {file_idx} AS file_idx, '{path.name}' AS source_file, "
            f"row_number() OVER () AS source_line, {cols} "
            f"FROM read_csv('{literal}', all_varchar=true, header=true)")
    conn.execute("CREATE OR REPLACE TEMP TABLE _raw AS " + " UNION ALL ".join(selects))
    return conn.execute("SELECT count(*) FROM _raw").fetchone()[0]


# Typed rows plus the reason each one is rejected (NULL = keep). Order of the CASE = rule order.
CLASSIFY_SQL = f"""
CREATE OR REPLACE TEMP TABLE _classified AS
WITH typed AS (
    SELECT
        file_idx, source_file, source_line,
        trim(invoice_no)                                   AS invoice_no,
        trim(stock_code)                                   AS stock_code,
        -- Strip outer whitespace and collapse internal runs ("MISO PRETTY  GUM"); blank -> NULL
        nullif(regexp_replace(trim(description), '\\s+', ' ', 'g'), '') AS description,
        try_cast(quantity AS INTEGER)                      AS quantity,
        try_cast(invoice_date AS TIMESTAMP)                AS invoice_ts,
        try_cast(unit_price AS DECIMAL(12, 3))             AS unit_price,
        -- Customer IDs are integers, but some exports write them as 13085.0
        try_cast(try_cast(customer_id AS DOUBLE) AS BIGINT) AS customer_id,
        trim(country)                                      AS country,
        quantity AS quantity_raw, invoice_date AS invoice_date_raw,
        unit_price AS unit_price_raw, customer_id AS customer_id_raw
    FROM _raw
),
first_file_per_invoice AS (
    SELECT invoice_no, min(file_idx) AS first_file_idx FROM typed GROUP BY invoice_no
),
flagged AS (
    SELECT
        t.*,
        upper(t.invoice_no) LIKE 'C%' AS is_cancellation,
        f.first_file_idx,
        row_number() OVER (
            PARTITION BY t.invoice_no, t.stock_code, t.description, t.quantity,
                         t.invoice_ts, t.unit_price, t.customer_id, t.country
            ORDER BY t.file_idx, t.source_line
        ) AS dup_rank
    FROM typed t JOIN first_file_per_invoice f USING (invoice_no)
)
SELECT
    *,
    CASE
        WHEN quantity IS NULL OR invoice_ts IS NULL OR unit_price IS NULL
             OR (customer_id IS NULL AND nullif(trim(customer_id_raw), '') IS NOT NULL)
            THEN 'unparseable_value'
        WHEN file_idx > first_file_idx               THEN 'cross_file_overlap'
        WHEN dup_rank > 1                            THEN 'exact_duplicate'
        WHEN upper(invoice_no) LIKE 'A%'             THEN 'bad_debt_adjustment'
        WHEN regexp_matches(stock_code, '{NON_PRODUCT_CODE_REGEX}', 'i') THEN 'non_product_code'
        WHEN customer_id IS NULL                     THEN 'missing_customer_id'
        WHEN unit_price <= 0                         THEN 'non_positive_price'
        WHEN quantity < 0 AND NOT is_cancellation    THEN 'non_cancel_negative'
        -- A 'C' invoice with a positive quantity contradicts itself (1 row in the real data)
        WHEN quantity > 0 AND is_cancellation        THEN 'cancel_positive_qty'
        WHEN quantity = 0                            THEN 'zero_quantity'
    END AS reject_reason
FROM flagged
"""

LINE_COLUMNS = """
    invoice_no, stock_code, description, quantity, invoice_ts, unit_price,
    customer_id, country, CAST(quantity * unit_price AS DECIMAL(14, 3)) AS line_total,
    source_file, source_line
"""

BUILD_TABLES_SQL = [
    f"""CREATE OR REPLACE TABLE raw_transactions AS
        SELECT {LINE_COLUMNS} FROM _classified
        WHERE reject_reason IS NULL AND NOT is_cancellation
        ORDER BY invoice_ts, invoice_no, source_file, source_line""",
    f"""CREATE OR REPLACE TABLE raw_returns AS
        SELECT {LINE_COLUMNS} FROM _classified
        WHERE reject_reason IS NULL AND is_cancellation
        ORDER BY invoice_ts, invoice_no, source_file, source_line""",
    """CREATE OR REPLACE TABLE audit_rejected_rows AS
        SELECT reject_reason, source_file, source_line,
               invoice_no, stock_code, description, quantity_raw AS quantity,
               invoice_date_raw AS invoice_date, unit_price_raw AS unit_price,
               customer_id_raw AS customer_id, country
        FROM _classified WHERE reject_reason IS NOT NULL
        ORDER BY reject_reason, source_file, source_line""",
    """CREATE OR REPLACE TABLE audit_load_summary AS
        SELECT
            CASE WHEN reject_reason IS NOT NULL THEN 'audit_rejected_rows'
                 WHEN is_cancellation THEN 'raw_returns'
                 ELSE 'raw_transactions' END            AS destination,
            coalesce(reject_reason, 'kept')             AS reason,
            count(*)                                    AS row_count,
            current_timestamp                           AS loaded_at
        FROM _classified GROUP BY ALL ORDER BY destination, row_count DESC""",
]


def validate(conn: duckdb.DuckDBPyConnection, n_input: int) -> None:
    """Integrity checks run inside the load transaction. Raises LoadError on any failure."""
    def scalar(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    n_out = scalar("SELECT (SELECT count(*) FROM raw_transactions) + (SELECT count(*) FROM raw_returns)"
                   " + (SELECT count(*) FROM audit_rejected_rows)")
    checks = {
        f"row conservation ({n_input:,} in vs {n_out:,} out)": n_input == n_out,
        "summary total matches input": scalar("SELECT sum(row_count) FROM audit_load_summary") == n_input,
        "raw_transactions not empty": scalar("SELECT count(*) FROM raw_transactions") > 0,
        "no null keys in raw_transactions": scalar(
            "SELECT count(*) FROM raw_transactions WHERE customer_id IS NULL OR invoice_no IS NULL "
            "OR stock_code IS NULL OR invoice_ts IS NULL") == 0,
        "sales have quantity > 0 and price > 0": scalar(
            "SELECT count(*) FROM raw_transactions WHERE quantity <= 0 OR unit_price <= 0") == 0,
        "no cancellations in raw_transactions": scalar(
            "SELECT count(*) FROM raw_transactions WHERE upper(invoice_no) LIKE 'C%'") == 0,
        "returns are 'C' invoices with quantity < 0": scalar(
            "SELECT count(*) FROM raw_returns WHERE upper(invoice_no) NOT LIKE 'C%' OR quantity >= 0") == 0,
        "no null customer_id in raw_returns": scalar(
            "SELECT count(*) FROM raw_returns WHERE customer_id IS NULL") == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise LoadError("Integrity checks failed: " + "; ".join(failed))


def load_and_clean_retail_data(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> dict[str, int]:
    """Run the full load. Returns {table: row_count} for the three data tables."""
    files = discover_csvs(Path(raw_dir))
    log.info("Loading %d file(s) into %s", len(files), db_path)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            n_input = stage_raw_files(conn, files)
            conn.execute(CLASSIFY_SQL)
            for sql in BUILD_TABLES_SQL:
                conn.execute(sql)
            validate(conn, n_input)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                  for t in ("raw_transactions", "raw_returns", "audit_rejected_rows")}
        log.info("Read %s raw rows", f"{n_input:,}")
        for destination, reason, n in conn.execute(
                "SELECT destination, reason, row_count FROM audit_load_summary").fetchall():
            log.info("  %-20s %-22s %10s", destination, reason, f"{n:,}")
        return counts
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        load_and_clean_retail_data(args.raw_dir, args.db_path)
    except (LoadError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
