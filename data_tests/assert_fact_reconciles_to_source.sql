-- The fact keeps every clean sale line: same row count and same revenue as the loader's table.
with src as (
    select count(*) as n, sum(line_total) as revenue from {{ source('raw', 'raw_transactions') }}
),
fct as (
    select count(*) as n, sum(line_total) as revenue from {{ ref('fct_invoice_lines') }}
)
select * from src, fct
where src.n <> fct.n or src.revenue <> fct.revenue
