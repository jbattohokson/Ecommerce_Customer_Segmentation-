# data/raw

Not committed. `pipelines/fetch_data.py` downloads the UCI Online Retail II dataset
(<https://archive.ics.uci.edu/dataset/502/online+retail+ii>, CC BY 4.0) and writes one CSV per
workbook sheet:

- `online_retail_II_2009_2010.csv`: 2009-12-01 to 2010-12-09
- `online_retail_II_2010_2011.csv`: 2010-12-01 to 2011-12-09

Columns as published: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`.
(The spec's `InvoiceNo` / `UnitPrice` / `CustomerID` are the names from the older Online Retail I release;
the loader renames them to snake_case either way.)
