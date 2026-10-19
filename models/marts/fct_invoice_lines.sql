-- Star-schema fact: one row per clean sale line, keyed to dim_customers, dim_products and dim_date.
select
    md5(concat_ws('|', source_file, cast(source_line as varchar))) as invoice_line_key,
    invoice_no,
    customer_id,
    stock_code,
    invoice_date,
    invoice_ts,
    quantity,
    unit_price,
    line_total
from {{ ref('stg_retail_transactions') }}
