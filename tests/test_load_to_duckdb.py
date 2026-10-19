"""Unit tests for pipelines/load_to_duckdb.py on small hand-built CSVs (no real data needed)."""

import duckdb
import pytest

from pipelines.load_to_duckdb import LoadError, load_and_clean_retail_data

HEADER = "Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country"

# Each line is built to trip exactly one rule (or none), named in the trailing comment.
FILE_1 = [
    "489434,85048,  GLASS BALL  LIGHTS ,12,2009-12-01 07:45:00,6.95,13085,United Kingdom",  # kept
    "489434,79323P,PINK CHERRY LIGHTS,12,2009-12-01 07:45:00,6.75,13085,United Kingdom",    # kept
    "489434,79323P,PINK CHERRY LIGHTS,12,2009-12-01 07:45:00,6.75,13085,United Kingdom",    # exact_duplicate
    "489435,22350,CAT BOWL,6,2009-12-01 08:00:00,2.10,,United Kingdom",                      # missing_customer_id
    "489436,POST,POSTAGE,1,2009-12-01 09:00:00,18.00,12345,Germany",                         # non_product_code
    "489437,22111,FREE SAMPLE,1,2009-12-01 10:00:00,0.00,12345,Germany",                     # non_positive_price
    "C489438,85048,GLASS BALL LIGHTS,-2,2009-12-02 10:00:00,6.95,13085,United Kingdom",      # return
    "489439,22222,DAMAGED,-5,2009-12-02 11:00:00,1.00,13085,United Kingdom",                 # non_cancel_negative
    "A506401,B,Adjust bad debt,1,2010-04-29 13:36:00,-53594.36,,United Kingdom",             # bad_debt_adjustment
    "489440,22333,BAD QTY,abc,2009-12-03 11:00:00,1.00,13085,United Kingdom",                # unparseable_value
    "536365,85123A,WHITE HEART,6,2010-12-01 08:26:00,2.55,17850.0,United Kingdom",           # kept (float id)
]
FILE_2 = [
    "536365,85123A,WHITE HEART,6,2010-12-01 08:26:00,2.55,17850,United Kingdom",             # cross_file_overlap
    "536366,71053,WHITE METAL LANTERN,6,2010-12-11 08:26:00,3.39,17850,United Kingdom",      # kept
]


def write(path, lines, header=HEADER):
    path.write_text("\n".join([header, *lines]) + "\n")


@pytest.fixture()
def loaded(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "a_2009.csv", FILE_1)
    write(raw / "b_2010.csv", FILE_2)
    db = tmp_path / "test.duckdb"
    counts = load_and_clean_retail_data(raw, db)
    conn = duckdb.connect(str(db), read_only=True)
    yield counts, conn
    conn.close()


def reasons(conn):
    return dict(conn.execute(
        "SELECT reason, row_count FROM audit_load_summary WHERE destination='audit_rejected_rows'"
    ).fetchall())


def test_counts_conserve_every_input_row(loaded):
    counts, _ = loaded
    assert counts == {"raw_transactions": 4, "raw_returns": 1, "audit_rejected_rows": 8}
    assert sum(counts.values()) == len(FILE_1) + len(FILE_2)


def test_each_reject_rule_fires_once(loaded):
    _, conn = loaded
    assert reasons(conn) == {
        "exact_duplicate": 1, "missing_customer_id": 1, "non_product_code": 1,
        "non_positive_price": 1, "non_cancel_negative": 1, "bad_debt_adjustment": 1,
        "unparseable_value": 1, "cross_file_overlap": 1,
    }


def test_descriptions_are_trimmed_and_collapsed(loaded):
    _, conn = loaded
    desc = conn.execute("SELECT description FROM raw_transactions WHERE stock_code='85048'").fetchone()[0]
    assert desc == "GLASS BALL LIGHTS"


def test_float_customer_ids_are_cast_to_integer(loaded):
    _, conn = loaded
    ids = {r[0] for r in conn.execute("SELECT customer_id FROM raw_transactions").fetchall()}
    assert ids == {13085, 17850}


def test_overlap_keeps_the_earlier_file(loaded):
    _, conn = loaded
    rows = conn.execute("SELECT source_file FROM raw_transactions WHERE invoice_no='536365'").fetchall()
    assert rows == [("a_2009.csv",)]


def test_returns_are_negative_with_negative_line_total(loaded):
    _, conn = loaded
    qty, total = conn.execute("SELECT quantity, line_total FROM raw_returns").fetchone()
    assert qty == -2 and float(total) == pytest.approx(-13.90)


def test_line_total_is_quantity_times_price(loaded):
    _, conn = loaded
    bad = conn.execute(
        "SELECT count(*) FROM raw_transactions WHERE line_total <> quantity * unit_price").fetchone()[0]
    assert bad == 0


def test_online_retail_i_header_names_are_accepted(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "old.csv", [FILE_1[0]],
          header="InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country")
    counts = load_and_clean_retail_data(raw, tmp_path / "x.duckdb")
    assert counts["raw_transactions"] == 1


def test_missing_column_raises_and_keeps_previous_load(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "good.csv", FILE_2)
    db = tmp_path / "x.duckdb"
    load_and_clean_retail_data(raw, db)

    write(raw / "good.csv", ["536366,71053,LANTERN,6,2010-12-11 08:26:00,3.39,United Kingdom"],
          header="Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Country")
    with pytest.raises(LoadError, match="customer_id"):
        load_and_clean_retail_data(raw, db)

    with duckdb.connect(str(db), read_only=True) as conn:
        # Loaded alone, FILE_2 has no earlier file to overlap with, so both rows were kept
        assert conn.execute("SELECT count(*) FROM raw_transactions").fetchone()[0] == 2


def test_empty_raw_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        load_and_clean_retail_data(tmp_path, tmp_path / "x.duckdb")
