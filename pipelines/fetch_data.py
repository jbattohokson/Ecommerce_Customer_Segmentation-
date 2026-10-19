"""Download the UCI Online Retail II dataset and write one CSV per workbook sheet.

UCI publishes the data as a single .xlsx inside a .zip (two sheets: "Year 2009-2010" and
"Year 2010-2011"). Parsing a 45 MB workbook takes a minute or two, so this runs once and every
later step reads the CSVs. Already-present CSVs are skipped unless --force is passed.

Usage:
    python pipelines/fetch_data.py            # download + convert if CSVs are missing
    python pipelines/fetch_data.py --force    # re-download and overwrite
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger("fetch_data")

DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
XLSX_NAME = "online_retail_II.xlsx"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Workbook sheet name -> CSV file name the loader expects.
SHEETS = {
    "Year 2009-2010": "online_retail_II_2009_2010.csv",
    "Year 2010-2011": "online_retail_II_2010_2011.csv",
}

# Read every column as text: Invoice and StockCode mix numbers and letters ("C489449", "85123A"),
# and Customer ID would otherwise become a float like 13085.0. The loader casts types itself.
EXPECTED_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "Price", "Customer ID", "Country"]


def download_zip(url: str = DATASET_URL, timeout: int = 120) -> bytes:
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        data = resp.read()
    log.info("Downloaded %.1f MB", len(data) / 1e6)
    return data


def convert_workbook(xlsx_bytes: bytes, out_dir: Path) -> list[Path]:
    """Write each known sheet of the workbook to a CSV in out_dir. Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    book = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=list(SHEETS), dtype=str,
                         engine="openpyxl")
    for sheet, csv_name in SHEETS.items():
        df = book[sheet]
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Sheet {sheet!r} is missing columns {sorted(missing)}; "
                             "the UCI file layout may have changed.")
        path = out_dir / csv_name
        df[EXPECTED_COLUMNS].to_csv(path, index=False)
        log.info("Wrote %s (%s rows)", path.name, f"{len(df):,}")
        written.append(path)
    return written


def fetch(out_dir: Path = RAW_DIR, force: bool = False) -> list[Path]:
    targets = [out_dir / name for name in SHEETS.values()]
    if not force and all(p.exists() for p in targets):
        log.info("CSVs already present in %s (use --force to re-download)", out_dir)
        return targets

    archive = zipfile.ZipFile(io.BytesIO(download_zip()))
    if XLSX_NAME not in archive.namelist():
        raise ValueError(f"{XLSX_NAME} not found in archive; contents: {archive.namelist()}")
    return convert_workbook(archive.read(XLSX_NAME), out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true", help="re-download even if CSVs exist")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetch(args.out_dir, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
