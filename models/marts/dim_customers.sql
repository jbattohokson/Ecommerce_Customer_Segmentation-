-- One row per customer with at least one clean sale. A handful of customers (12) ordered
-- from more than one country; the country with the most sale lines (latest as tie-break) wins.
with country_rank as (
    select
        customer_id,
        country,
        row_number() over (
            partition by customer_id order by count(*) desc, max(invoice_ts) desc, country
        ) as rn
    from {{ ref('stg_retail_transactions') }}
    group by customer_id, country
)
select
    t.customer_id,
    c.country,
    c.country = 'United Kingdom'       as is_domestic,
    min(t.invoice_date)                as first_purchase_date,
    max(t.invoice_date)                as last_purchase_date,
    -- Acquisition cohort for cohort analysis in the app
    cast(date_trunc('month', min(t.invoice_date)) as date) as cohort_month
from {{ ref('stg_retail_transactions') }} t
join country_rank c on c.customer_id = t.customer_id and c.rn = 1
group by t.customer_id, c.country
