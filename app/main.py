"""Streamlit app: executive cohort dashboard, local-AI campaign generator, churn-risk simulator.

Run from the repo root:
    streamlit run app/main.py

Reads retail_analytics.duckdb read-only (loader -> dbt -> ml/segmentation.py must have run).
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# `streamlit run app/main.py` puts app/ on sys.path, not the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import retention_agent as ra  # noqa: E402
from ml.segmentation import CHURN_RISK_PERSONAS, SEGMENT_ORDER, load_model_from_db  # noqa: E402

DB_PATH = ra.DB_PATH

st.set_page_config(page_title="Retail Retention Engine", page_icon="🛍️", layout="wide")

# ---- Colour: one fixed hue per segment (validated categorical slots 1-4), never by rank -------
IS_DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, (
    ["#3987e5", "#d95926", "#199e70", "#c98500"] if IS_DARK
    else ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])))
NEUTRAL = "#8a8983"
SEQUENTIAL = "Blues"
PLOT_LAYOUT = dict(margin=dict(l=8, r=8, t=48, b=8), legend_title_text="", hovermode="closest",
                   bargap=0.25)


def style(fig: go.Figure, **overrides) -> go.Figure:
    """Shared layout; per-chart overrides win (e.g. hovermode='x unified' on time series)."""
    return fig.update_layout(**{**PLOT_LAYOUT, **overrides})


# ---- Data access ------------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    # Short-lived read-only connections so the app never blocks a pipeline refresh.
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(sql, list(params)).df()


def tables_ready() -> list[str]:
    needed = {"fct_customer_transactions", "agg_monthly_sales", "dim_customer_segments",
              "ml_segment_profiles", "ml_model_params"}
    if not DB_PATH.exists():
        return sorted(needed)
    have = set(query("SELECT table_name FROM information_schema.tables WHERE table_schema='marts'")
               ["table_name"])
    return sorted(needed - have)


@st.cache_resource(show_spinner=False)
def segment_model():
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return load_model_from_db(conn)


def gbp(x: float) -> str:
    return f"£{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"£{x/1e3:,.1f}k" if abs(x) >= 1e4 else f"£{x:,.0f}"


def ordered(df: pd.DataFrame, col: str = "segment_name") -> pd.DataFrame:
    order = {s: i for i, s in enumerate(SEGMENT_ORDER)}
    return df.assign(_o=df[col].map(order).fillna(99)).sort_values("_o").drop(columns="_o")


# ---- Header / empty state ---------------------------------------------------------------------

st.title("E-Commerce Customer Segmentation & AI Retention Engine")
st.caption("UCI Online Retail II · UK online giftware retailer · Dec 2009 – Dec 2011 · "
           "RFM K-Means segments · local CrewAI + Ollama copywriter")

missing = tables_ready()
if missing:
    st.error(f"The warehouse isn't built yet (missing: {', '.join(missing)}). From the repo root run:")
    st.code("python pipelines/fetch_data.py\npython pipelines/load_to_duckdb.py\n"
            "DBT_PROFILES_DIR=. dbt build\npython ml/segmentation.py", language="bash")
    st.stop()

profiles = ordered(query("SELECT * FROM marts.ml_segment_profiles"))
monthly = query("SELECT * FROM marts.agg_monthly_sales ORDER BY month_start")
customers = query("""
    SELECT s.customer_id, s.segment_name, s.is_churn_risk, s.recency_days, s.frequency,
           s.monetary, t.avg_order_value, t.gross_spend
    FROM marts.dim_customer_segments s JOIN marts.fct_customer_transactions t USING (customer_id)""")
meta = query("""
    SELECT (SELECT max(segmented_at) FROM marts.dim_customer_segments) AS segmented_at,
           (SELECT max(loaded_at) FROM main.audit_load_summary)          AS loaded_at,
           (SELECT k FROM marts.ml_k_selection WHERE is_chosen)           AS k,
           (SELECT silhouette FROM marts.ml_k_selection WHERE is_chosen)  AS silhouette""").iloc[0]

with st.sidebar:
    st.subheader("Pipeline status")
    st.write(f"**Loaded:** {pd.Timestamp(meta.loaded_at):%Y-%m-%d %H:%M}")
    st.write(f"**Segmented:** {pd.Timestamp(meta.segmented_at):%Y-%m-%d %H:%M}")
    st.write(f"**Model:** K-Means, k={int(meta.k)}, silhouette {meta.silhouette:.3f}")
    st.write(f"**Active customers:** {len(customers):,}")
    st.caption("RFM snapshot date is 2011-12-10, the day after the last invoice. "
               "'Active' means net spend > 0 (20 customers who returned everything are excluded).")
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

tab1, tab2, tab3 = st.tabs(["📊 Executive Cohort & LTV", "🤖 AI Retention Campaigns",
                            "⚙️ Churn Risk Simulator"])

# =============================================================================================
# Tab 1: Executive cohort & LTV dashboard
# =============================================================================================
with tab1:
    risk = profiles[profiles.is_churn_risk]
    total_net = customers["monetary"].sum()
    full_months = monthly[~monthly.is_partial_month]
    last12 = full_months.tail(12)["net_revenue"].sum()
    prior12 = full_months.iloc[-24:-12]["net_revenue"].sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Active customers", f"{len(customers):,}")
    k2.metric("Net revenue (2 yrs)", gbp(total_net))
    k3.metric("Net revenue, last 12 full months", gbp(last12),
              f"{(last12 / prior12 - 1):+.1%} vs prior 12" if prior12 else None)
    k4.metric("Avg order value", gbp(customers["gross_spend"].sum() / customers["frequency"].sum()))
    k5.metric(f"Churn-risk LTV ({risk['customers'].sum():,} customers)", gbp(risk["total_monetary"].sum()),
              help="Net lifetime spend of customers in the At-Risk Whales and Dormant Buyers segments.")

    c1, c2 = st.columns(2)
    with c1:
        share = profiles.melt(id_vars="segment_name", value_vars=["customer_share", "revenue_share"],
                              var_name="measure", value_name="share")
        share["measure"] = share["measure"].map({"customer_share": "Share of customers",
                                                 "revenue_share": "Share of net revenue"})
        fig = px.bar(share, x="share", y="measure", color="segment_name", orientation="h",
                     color_discrete_map=SEGMENT_COLORS, text=share["share"].map("{:.0%}".format),
                     category_orders={"segment_name": SEGMENT_ORDER},
                     title="Revenue concentration: who the revenue comes from")
        fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)",
                          textposition="inside", insidetextanchor="middle",
                          hovertemplate="%{fullData.name}<br>%{y}: %{x:.1%}<extra></extra>")
        style(fig, barmode="stack", xaxis_tickformat=".0%",
                          xaxis_title=None, yaxis_title=None, legend_orientation="h", legend_y=-0.15)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(profiles, x="segment_name", y="customers", color="segment_name",
                     color_discrete_map=SEGMENT_COLORS, text="customers",
                     custom_data=["median_monetary", "median_frequency", "median_recency_days"],
                     title="Segment size (customers)")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False,
                          hovertemplate="<b>%{x}</b><br>%{y:,} customers<br>Median spend £%{customdata[0]:,.0f}"
                                        "<br>Median orders %{customdata[1]:.0f}"
                                        "<br>Median days since last order %{customdata[2]:.0f}<extra></extra>")
        style(fig, showlegend=False, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### What each segment looks like (medians)")
    m1, m2, m3 = st.columns(3)
    for col, metric, title, fmt in [
        (m1, "median_recency_days", "Days since last order", "%{x:.0f}"),
        (m2, "median_frequency", "Orders", "%{x:.0f}"),
        (m3, "median_monetary", "Net spend (£)", "£%{x:,.0f}"),
    ]:
        fig = px.bar(profiles, x=metric, y="segment_name", orientation="h", color="segment_name",
                     color_discrete_map=SEGMENT_COLORS, title=title,
                     category_orders={"segment_name": SEGMENT_ORDER})
        fig.update_traces(texttemplate=fmt, textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}: " + fmt + "<extra></extra>")
        style(fig, showlegend=False, height=240, xaxis_title=None, yaxis_title=None,
              xaxis_visible=False, xaxis_range=[0, profiles[metric].max() * 1.3])
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Sales velocity")
    v1, v2 = st.columns(2)
    with v1:
        fig = go.Figure(go.Scatter(
            x=monthly.month_start, y=monthly.net_revenue, mode="lines+markers",
            line=dict(width=2, color=SEGMENT_COLORS["Loyal Champions"]), marker=dict(size=8),
            hovertemplate="%{x|%b %Y}<br>Net revenue £%{y:,.0f}<extra></extra>"))
        last = monthly.iloc[-1]
        if last.is_partial_month:
            fig.add_annotation(x=last.month_start, y=last.net_revenue, text="Dec 2011: 9 days only",
                               showarrow=True, arrowhead=0, ax=-60, ay=-30)
        style(fig, title="Net revenue by month (£)", hovermode="x unified",
                          yaxis_tickprefix="£", yaxis_tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        nr = monthly.melt(id_vars="month_start", value_vars=["returning_customers", "new_customers"],
                          var_name="type", value_name="customers")
        nr["type"] = nr["type"].map({"returning_customers": "Returning", "new_customers": "New"})
        fig = px.bar(nr, x="month_start", y="customers", color="type",
                     color_discrete_map={"Returning": SEGMENT_COLORS["Loyal Champions"],
                                         "New": SEGMENT_COLORS["Recent Newcomers"]},
                     title="Active customers by month: new vs returning")
        fig.update_traces(marker_line_width=0, hovertemplate="%{x|%b %Y}<br>%{fullData.name}: %{y:,}<extra></extra>")
        style(fig, barmode="stack", xaxis_title=None, yaxis_title=None,
                          legend_orientation="h", legend_y=-0.15, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Dec 2009 counts every customer as 'new' because the data starts there "
               "(left-censoring), so new-customer counts are only meaningful from early 2010.")

    with st.expander("Segment table and model diagnostics"):
        st.dataframe(profiles[["segment_name", "is_churn_risk", "customers", "customer_share",
                               "revenue_share", "median_recency_days", "median_frequency",
                               "median_monetary", "total_monetary"]]
                     .style.format({"customer_share": "{:.1%}", "revenue_share": "{:.1%}",
                                    "median_monetary": "£{:,.0f}", "total_monetary": "£{:,.0f}",
                                    "median_frequency": "{:.0f}", "median_recency_days": "{:.0f}"}),
                     hide_index=True, use_container_width=True)
        ks = query("SELECT * FROM marts.ml_k_selection ORDER BY k")
        fig = go.Figure(go.Scatter(
            x=ks.k, y=ks.silhouette, mode="lines+markers", line=dict(width=2, color=NEUTRAL),
            marker=dict(size=10, color=[SEGMENT_COLORS["Loyal Champions"] if c else NEUTRAL
                                        for c in ks.is_chosen]),
            hovertemplate="k=%{x}<br>silhouette %{y:.3f}<extra></extra>"))
        style(fig, height=260, xaxis_title="k (clusters)",
                          yaxis_title="Silhouette", title="Choosing k: silhouette on log1p-scaled RFM "
                          "(highlighted = chosen, from k ∈ {4, 5, 6})")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================================
# Tab 2: AI retention campaign generator
# =============================================================================================
with tab2:
    st.subheader("Retention campaign generator (local AI)")
    st.caption(f"A CrewAI 'Lifecycle Marketing Analyst' on Ollama `{ra.OLLAMA_MODEL}` pulls the "
               "cohort brief from DuckDB through fixed read-only tools, then drafts the campaign. "
               "Nothing leaves this machine. A run takes 1–3 minutes on a laptop.")

    seg_options = sorted(profiles.segment_name, key=lambda s: (s not in CHURN_RISK_PERSONAS,
                                                               SEGMENT_ORDER.index(s) if s in SEGMENT_ORDER else 99))
    segment = st.selectbox("Segment", seg_options,
                           format_func=lambda s: f"{s}  ⚠ churn risk" if s in CHURN_RISK_PERSONAS else s)
    brief = ra.segment_brief(segment)
    policy = brief["incentive_policy"]

    b1, b2, b3, b4 = st.columns(4)
    b1.metric(f"Customers ({brief['share_of_all_customers_pct']}% of base)", f"{brief['customers']:,}")
    b2.metric("Share of net revenue", f"{brief['share_of_total_net_revenue_pct']}%")
    b3.metric("Median days since last order", f"{brief['median_days_since_last_order']:.0f}")
    b4.metric("Discount cap (policy)", f"{policy['max_discount_pct']}%")
    st.write(f"**Objective:** {policy['objective']}  \n**Preferred incentive:** {policy['preferred_incentive']}")
    st.dataframe(pd.DataFrame(brief["top_products"]).rename(columns={
        "product": "Top products", "customers_who_bought": "Customers who bought", "revenue": "Revenue (£)"}),
        hide_index=True, use_container_width=True)

    try:
        ra.check_ollama()
        ollama_ok, ollama_msg = True, ""
    except ra.AgentSetupError as exc:
        ollama_ok, ollama_msg = False, str(exc)
    if not ollama_ok:
        st.warning(ollama_msg)

    if st.button("Generate AI retention campaign", type="primary", disabled=not ollama_ok):
        with st.spinner("Local agent is analysing the cohort and writing copy…"):
            try:
                st.session_state[f"campaign::{segment}"] = ra.generate_retention_campaign(segment)
            except Exception as exc:  # surface LLM/tool failures in the UI rather than crashing
                st.error(f"Campaign generation failed: {exc}")

    campaign = st.session_state.get(f"campaign::{segment}")
    saved = sorted(ra.CAMPAIGN_DIR.glob("*.md"), reverse=True) if ra.CAMPAIGN_DIR.exists() else []
    if campaign is None and saved:
        slug = segment.lower().replace("-", "_").replace(" ", "_")
        latest = next((p for p in saved if p.name.startswith(slug)), None)
        if latest:
            st.info(f"Showing the most recent saved campaign for this segment ({latest.name}).")
            campaign = latest.read_text()

    if campaign:
        problems = ra.campaign_problems(campaign, segment)
        if problems:
            st.error("Guardrail: " + "; ".join(problems) + ". Regenerate or edit before sending.")
        else:
            st.success(f"Guardrail: all sections present, merge fields used, no discount above "
                       f"the {policy['max_discount_pct']}% cap.")
        with st.container(border=True):
            st.markdown(campaign)
        st.download_button("Download markdown", campaign,
                           file_name=f"campaign_{segment.lower().replace(' ', '_')}.md")

# =============================================================================================
# Tab 3: Churn risk simulator
# =============================================================================================
with tab3:
    st.subheader("Churn risk simulator")
    left, right = st.columns([1, 2])
    with left:
        threshold = st.slider("Lapse threshold: a customer counts as lapsed after this many days "
                              "without an order", 30, 365, 90, step=15)
        fast_forward = st.slider("Fast-forward: days with no new orders from anyone", 0, 365, 90, step=15)
        st.caption("The threshold re-labels customers by a simple rule. Fast-forward ages every "
                   "customer's recency and re-scores them with the trained K-Means model "
                   "(same scaler and centroids), so you can see who drifts into a churn-risk "
                   "segment if nothing is done.")

    lapsed = customers["recency_days"] > threshold
    by_seg = (customers.assign(status=np.where(lapsed, "Lapsed", "Active"))
              .groupby(["segment_name", "status"])
              .agg(customers=("customer_id", "size"), value=("monetary", "sum")).reset_index())
    with right:
        s1, s2, s3 = st.columns(3)
        s1.metric(f"Lapsed > {threshold}d ({lapsed.mean():.0%} of base)", f"{lapsed.sum():,}")
        s2.metric("Lapsed customers' LTV", gbp(customers.loc[lapsed, "monetary"].sum()))
        champs_lapsed = customers[(customers.segment_name == "Loyal Champions") & lapsed]
        s3.metric(f"Lapsed Champions ({gbp(champs_lapsed['monetary'].sum())} LTV)",
                  f"{len(champs_lapsed):,}")
        fig = px.bar(ordered(by_seg), x="customers", y="segment_name", color="status", orientation="h",
                     color_discrete_map={"Active": SEGMENT_COLORS["Loyal Champions"], "Lapsed": NEUTRAL},
                     text="customers", category_orders={"segment_name": SEGMENT_ORDER,
                                                        "status": ["Active", "Lapsed"]},
                     title=f"Active vs lapsed by segment at a {threshold}-day threshold")
        fig.update_traces(texttemplate="%{text:,}", textposition="inside",
                          hovertemplate="%{y}<br>%{fullData.name}: %{x:,}<extra></extra>")
        style(fig, barmode="stack", xaxis_title=None, yaxis_title=None,
                          legend_orientation="h", legend_y=-0.15)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"##### If no one orders for {fast_forward} days")
    model = segment_model()
    aged = customers[["customer_id", "recency_days", "frequency", "monetary", "segment_name"]].copy()
    aged["recency_days"] = aged["recency_days"] + fast_forward
    aged["segment_after"] = model.predict_names(aged)
    before = customers["segment_name"].value_counts()
    after = aged["segment_after"].value_counts()
    sizes = pd.DataFrame({"Today": before, f"After {fast_forward} days": after}).fillna(0).astype(int)
    sizes = sizes.reindex([s for s in SEGMENT_ORDER if s in sizes.index]
                          + [s for s in sizes.index if s not in SEGMENT_ORDER])

    risk_now = customers["segment_name"].isin(CHURN_RISK_PERSONAS)
    risk_after = aged["segment_after"].isin(CHURN_RISK_PERSONAS)
    newly = aged[risk_after & ~risk_now]
    f1, f2, f3 = st.columns(3)
    f1.metric("Customers in churn-risk segments", f"{risk_after.sum():,}",
              f"{risk_after.sum() - risk_now.sum():+,} vs today", delta_color="inverse")
    f2.metric("Lifetime value moving into churn risk", gbp(newly["monetary"].sum()))
    f3.metric("Loyal Champions remaining", f"{(aged.segment_after == 'Loyal Champions').sum():,}",
              f"{(aged.segment_after == 'Loyal Champions').sum() - (customers.segment_name == 'Loyal Champions').sum():+,}",
              delta_color="normal")

    g1, g2 = st.columns(2)
    with g1:
        long = sizes.reset_index(names="segment").melt(id_vars="segment", var_name="when",
                                                       value_name="customers")
        fig = px.bar(long, x="customers", y="when", color="segment", orientation="h",
                     color_discrete_map=SEGMENT_COLORS, text="customers",
                     category_orders={"segment": SEGMENT_ORDER}, title="Segment sizes, today vs after")
        fig.update_traces(texttemplate="%{text:,}", textposition="inside",
                          hovertemplate="%{fullData.name}<br>%{y}: %{x:,}<extra></extra>")
        style(fig, barmode="stack", xaxis_title=None, yaxis_title=None,
                          legend_orientation="h", legend_y=-0.2)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        moves = pd.crosstab(customers["segment_name"], aged["segment_after"])
        order = [s for s in SEGMENT_ORDER if s in moves.index or s in moves.columns]
        moves = moves.reindex(index=order, columns=order, fill_value=0)
        fig = px.imshow(moves, text_auto=",", color_continuous_scale=SEQUENTIAL, aspect="auto",
                        labels=dict(x="Segment after", y="Segment today", color="Customers"),
                        title="Where customers move")
        fig.update_traces(hovertemplate="%{y} → %{x}: %{z:,}<extra></extra>", xgap=2, ygap=2)
        style(fig, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
