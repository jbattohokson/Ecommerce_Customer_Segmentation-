{#
  Use custom schema names exactly as written ("staging", "marts") instead of
  dbt's default "<target_schema>_<custom>" (e.g. "main_marts"). The ML step, AI agent
  and the Streamlit app query marts.<table>, so the names should be clean.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
