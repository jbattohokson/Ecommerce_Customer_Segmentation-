{# The RFM snapshot date, as a DATE literal (see vars.snapshot_date in dbt_project.yml). #}
{% macro snapshot_date() -%}
    cast('{{ var("snapshot_date") }}' as date)
{%- endmacro %}
