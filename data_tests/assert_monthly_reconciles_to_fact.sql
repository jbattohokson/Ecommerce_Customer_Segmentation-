-- The monthly series sums to total fact revenue, and new customers sum to the customer count.
with m as (
    select sum(gross_revenue) as revenue, sum(new_customers) as new_customers
    from {{ ref('agg_monthly_sales') }}
)
select * from m
where m.revenue <> (select sum(line_total) from {{ ref('fct_invoice_lines') }})
   or m.new_customers <> (select count(*) from {{ ref('dim_customers') }})
