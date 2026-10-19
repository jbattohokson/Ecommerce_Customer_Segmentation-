-- Recency is consistent with dim_customers' last purchase date and the snapshot var.
select c.customer_id
from {{ ref('fct_customer_transactions') }} f
join {{ ref('dim_customers') }} c using (customer_id)
where f.recency_days <> date_diff('day', c.last_purchase_date, {{ snapshot_date() }})
