-- One row per cancelled product line. Quantity and line_total are negative.
select
    invoice_no,
    stock_code,
    description,
    customer_id,
    country,
    invoice_ts,
    cast(invoice_ts as date)           as return_date,
    quantity,
    unit_price,
    line_total,
    source_file,
    source_line
from {{ source('raw', 'raw_returns') }}
