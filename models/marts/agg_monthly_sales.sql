-- One row per calendar month: the sales-velocity time series for the dashboard.
-- Dec 2011 is a partial month (data ends 2011-12-09); is_partial_month flags it.
with lines as (
    select t.*, d.month_start
    from {{ ref('fct_invoice_lines') }} t
    join {{ ref('dim_date') }} d on d.date_day = t.invoice_date
),
first_orders as (
    select customer_id, cohort_month from {{ ref('dim_customers') }}
),
monthly as (
    select
        month_start,
        sum(line_total)                     as gross_revenue,
        count(distinct invoice_no)          as orders,
        count(distinct customer_id)         as active_customers,
        sum(quantity)                       as units_sold
    from lines
    group by month_start
),
returns as (
    select cast(date_trunc('month', return_date) as date) as month_start,
           -sum(line_total) as returned_value
    from {{ ref('stg_retail_returns') }}
    group by 1
),
new_customers as (
    select cohort_month as month_start, count(*) as new_customers
    from first_orders group by 1
)
select
    m.month_start,
    m.gross_revenue,
    coalesce(r.returned_value, 0)                       as returned_value,
    m.gross_revenue - coalesce(r.returned_value, 0)     as net_revenue,
    m.orders,
    m.active_customers,
    coalesce(n.new_customers, 0)                        as new_customers,
    m.active_customers - coalesce(n.new_customers, 0)   as returning_customers,
    m.units_sold,
    round(m.gross_revenue / m.orders, 2)                as avg_order_value,
    -- Partial if the last data day (snapshot - 1) falls before the month's final day.
    m.month_start = (select max(month_start) from monthly)
        and {{ snapshot_date() }} - 1 < last_day(m.month_start) as is_partial_month
from monthly m
left join returns r using (month_start)
left join new_customers n using (month_start)
order by m.month_start
