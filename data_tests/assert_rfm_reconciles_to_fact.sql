-- Customer-level RFM rolls up exactly to the fact: total gross spend and total orders match.
with rfm as (
    select sum(gross_spend) as spend, sum(frequency) as orders, count(*) as customers
    from {{ ref('fct_customer_transactions') }}
),
fct as (
    select sum(line_total) as spend, count(distinct invoice_no) as orders,
           count(distinct customer_id) as customers
    from {{ ref('fct_invoice_lines') }}
)
select * from rfm, fct
where rfm.spend <> fct.spend or rfm.orders <> fct.orders or rfm.customers <> fct.customers
