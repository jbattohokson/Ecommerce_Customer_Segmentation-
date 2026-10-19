{#
  Generic test: every non-null value of the column falls within [min_value, max_value].
  Either bound can be omitted. Written here so the project needs no dbt packages.
#}
{% test accepted_range(model, column_name, min_value=none, max_value=none) %}
select {{ column_name }}
from {{ model }}
where {{ column_name }} is not null
  and (
      false
      {% if min_value is not none %} or {{ column_name }} < {{ min_value }} {% endif %}
      {% if max_value is not none %} or {{ column_name }} > {{ max_value }} {% endif %}
  )
{% endtest %}
