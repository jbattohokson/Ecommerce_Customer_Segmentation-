-- One row per customer: the RFM base the K-Means step (ml/segmentation.py) clusters on.
--   recency_days  days from the last purchase date to the snapshot date (var snapshot_date)
--   frequency     distinct sale invoices (orders), not lines
--   monetary      net spend: gross sales minus the value of returns
-- Returns by customers with no sale in the window (they bought before Dec 2009) can't be
-- attributed to an order here, so they're left out rather than producing negative customers.
with sales as (
    select
        customer_id,
        count(distinct invoice_no)                  as frequency,
        sum(line_total)                             as gross_spend,
        sum(quantity)                               as units_purchased,
        count(distinct stock_code)                  as distinct_products,
        min(invoice_date)                           as first_purchase_date,
        max(invoice_date)                           as last_purchase_date,
        count(distinct date_trunc('month', invoice_date)) as active_months
    from {{ ref('stg_retail_transactions') }}
    group by customer_id
),
returns as (
    select
        customer_id,
        count(distinct invoice_no)                  as return_invoices,
        -sum(line_total)                            as returned_value,
        -sum(quantity)                              as units_returned
    from {{ ref('stg_retail_returns') }}
    group by customer_id
)
select
    s.customer_id,
    s.first_purchase_date,
    s.last_purchase_date,
    date_diff('day', s.last_purchase_date, {{ snapshot_date() }})   as recency_days,
    date_diff('day', s.first_purchase_date, {{ snapshot_date() }})  as tenure_days,
    s.frequency,
    s.gross_spend,
    coalesce(r.returned_value, 0)                                    as returned_value,
    s.gross_spend - coalesce(r.returned_value, 0)                    as monetary,
    round(s.gross_spend / s.frequency, 2)                            as avg_order_value,
    s.units_purchased,
    coalesce(r.units_returned, 0)                                    as units_returned,
    coalesce(r.return_invoices, 0)                                   as return_invoices,
    round(coalesce(r.returned_value, 0) / s.gross_spend, 4)          as return_rate,
    s.distinct_products,
    s.active_months
from sales s
left join returns r using (customer_id)
