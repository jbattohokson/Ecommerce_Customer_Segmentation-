-- One row per clean sale line. Cleaning already happened in the loader; this view only
-- renames, adds the calendar date, and keeps the grain explicit.
select
    invoice_no,
    stock_code,
    description,
    customer_id,
    country,
    invoice_ts,
    cast(invoice_ts as date)           as invoice_date,
    quantity,
    unit_price,
    line_total,
    source_file,
    source_line
from {{ source('raw', 'raw_transactions') }}
