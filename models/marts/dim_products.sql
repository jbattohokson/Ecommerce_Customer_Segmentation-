-- One row per stock code. 594 codes appear with more than one description over the two years
-- (typos, renamed listings), so the most frequently used description is taken as canonical.
with descriptions as (
    select
        stock_code,
        description,
        count(*) as n_lines,
        row_number() over (partition by stock_code order by count(*) desc, description) as rn
    from {{ ref('stg_retail_transactions') }}
    where description is not null
    group by stock_code, description
),
stats as (
    select
        stock_code,
        count(*)                        as n_sale_lines,
        sum(quantity)                   as units_sold,
        sum(line_total)                 as gross_revenue,
        median(unit_price)              as median_unit_price,
        count(distinct customer_id)     as n_customers,
        min(invoice_date)               as first_sold_date,
        max(invoice_date)               as last_sold_date
    from {{ ref('stg_retail_transactions') }}
    group by stock_code
)
select
    s.stock_code,
    coalesce(d.description, '(no description)') as description,
    s.n_sale_lines,
    s.units_sold,
    s.gross_revenue,
    s.median_unit_price,
    s.n_customers,
    s.first_sold_date,
    s.last_sold_date
from stats s
left join descriptions d on d.stock_code = s.stock_code and d.rn = 1
