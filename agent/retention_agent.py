"""Local CrewAI retention copywriter: cohort data from DuckDB -> email campaign, via Ollama.

One "Lifecycle Marketing Analyst" agent runs two sequential tasks:
  1. Cohort analysis: call the DuckDB tools to profile a segment (size, RFM medians, revenue
     share, top products, top countries, how long since they last ordered) and summarise it.
  2. Campaign copy: turn that summary into a ready-to-send markdown email campaign with a
     discount/incentive that stays inside the segment's INCENTIVE_POLICY.

Design choices:
  - The tools run fixed, parameterised SQL rather than free-form queries written by the model.
    Small local models write unreliable SQL, and fixed queries mean every number in the email
    traces back to the warehouse. The connection is read-only either way.
  - Incentive limits come from INCENTIVE_POLICY (a business rule), not from the model, so the
    agent can't promise a 50%-off deal to customers who'd buy anyway.
  - Everything runs locally: no API key, no cost, no customer data leaves the machine.

Configuration (environment variables):
    OLLAMA_MODEL      default "llama3.1" (any pulled Ollama model, e.g. "llama3.2", "mistral")
    OLLAMA_BASE_URL   default "http://localhost:11434"
    RETAIL_DB_PATH    default "<repo>/retail_analytics.duckdb"

Usage:
    python agent/retention_agent.py --segment "At-Risk Whales"
    python agent/retention_agent.py --all-risk          # every churn-risk segment
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import duckdb
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

log = logging.getLogger("retention_agent")

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("RETAIL_DB_PATH", REPO_ROOT / "retail_analytics.duckdb"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CAMPAIGN_DIR = REPO_ROOT / "campaigns"

# Business guardrails per segment: what the campaign is for and the most it may offer.
INCENTIVE_POLICY = {
    "At-Risk Whales": {
        "objective": "win back lapsing high-value accounts before they churn for good",
        "max_discount_pct": 15,
        "preferred_incentive": "VIP perk (free shipping on the next 3 orders or early access to new "
                               "ranges) first; a percentage discount only as a secondary hook",
    },
    "Dormant Buyers": {
        "objective": "reactivate one-time or long-lapsed buyers at low cost",
        "max_discount_pct": 20,
        "preferred_incentive": "one-time percentage discount with a clear 14-day expiry",
    },
    "Recent Newcomers": {
        "objective": "convert recent first/second-time buyers into repeat customers",
        "max_discount_pct": 10,
        "preferred_incentive": "second-order discount or bundle suggestion based on what they bought",
    },
    "Loyal Champions": {
        "objective": "reward and retain the best customers without training them to wait for discounts",
        "max_discount_pct": 5,
        "preferred_incentive": "non-discount recognition: loyalty tier, thank-you gift, referral bonus",
    },
}
DEFAULT_POLICY = {"objective": "re-engage this cohort", "max_discount_pct": 10,
                  "preferred_incentive": "modest one-time discount"}


class AgentSetupError(RuntimeError):
    """Raised when Ollama or the database isn't ready, with the command that fixes it."""


# --------------------------------------------------------------------------------------------
# Data access (plain functions so the app and tests can call them without an LLM)
# --------------------------------------------------------------------------------------------

def _query(sql: str, params: list | None = None) -> list[dict]:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        cur = conn.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def list_segments() -> list[dict]:
    return _query("""
        SELECT segment_name, is_churn_risk, customers, customer_share, revenue_share,
               median_recency_days, median_frequency, round(median_monetary, 2) AS median_monetary
        FROM marts.ml_segment_profiles ORDER BY cluster_id""")


def segment_brief(segment_name: str, top_n: int = 5) -> dict:
    """Everything the copywriter needs about one segment, as plain JSON-able data."""
    profile = _query("SELECT * FROM marts.ml_segment_profiles WHERE segment_name = ?", [segment_name])
    if not profile:
        valid = [s["segment_name"] for s in list_segments()]
        raise ValueError(f"Unknown segment {segment_name!r}. Valid segments: {valid}")
    p = profile[0]

    top_products = _query("""
        SELECT p.description AS product, count(DISTINCT f.customer_id) AS customers_who_bought,
               round(sum(f.line_total), 2) AS revenue
        FROM marts.fct_invoice_lines f
        JOIN marts.dim_customer_segments s USING (customer_id)
        JOIN marts.dim_products p USING (stock_code)
        WHERE s.segment_name = ?
        GROUP BY p.description ORDER BY customers_who_bought DESC, revenue DESC LIMIT ?""",
        [segment_name, top_n])

    top_countries = _query("""
        SELECT c.country, count(*) AS customers
        FROM marts.dim_customer_segments s JOIN marts.dim_customers c USING (customer_id)
        WHERE s.segment_name = ? GROUP BY c.country ORDER BY customers DESC LIMIT 3""",
        [segment_name])

    behaviour = _query("""
        SELECT round(avg(t.avg_order_value), 2) AS avg_order_value,
               round(avg(t.return_rate) * 100, 1) AS avg_return_rate_pct,
               round(avg(t.distinct_products), 1) AS avg_distinct_products,
               count(*) FILTER (WHERE t.recency_days <= 90)  AS ordered_last_90d,
               count(*) FILTER (WHERE t.recency_days BETWEEN 91 AND 180) AS last_order_91_180d,
               count(*) FILTER (WHERE t.recency_days BETWEEN 181 AND 365) AS last_order_181_365d,
               count(*) FILTER (WHERE t.recency_days > 365) AS last_order_over_1y
        FROM marts.dim_customer_segments s
        JOIN marts.fct_customer_transactions t USING (customer_id)
        WHERE s.segment_name = ?""", [segment_name])[0]

    return {
        "segment_name": segment_name,
        "is_churn_risk": bool(p["is_churn_risk"]),
        "customers": int(p["customers"]),
        "share_of_all_customers_pct": round(100 * p["customer_share"], 1),
        "share_of_total_net_revenue_pct": round(100 * p["revenue_share"], 1),
        "median_days_since_last_order": float(p["median_recency_days"]),
        "median_orders": float(p["median_frequency"]),
        "median_net_spend_gbp": round(float(p["median_monetary"]), 2),
        "total_net_spend_gbp": round(float(p["total_monetary"]), 2),
        **{k: (float(v) if v is not None else None) for k, v in behaviour.items()},
        "top_products": [{**r, "revenue": float(r["revenue"])} for r in top_products],
        "top_countries": top_countries,
        "incentive_policy": INCENTIVE_POLICY.get(segment_name, DEFAULT_POLICY),
    }


# --------------------------------------------------------------------------------------------
# CrewAI tools (thin wrappers: the docstring is what the model sees)
# --------------------------------------------------------------------------------------------

@tool("List customer segments")
def list_segments_tool() -> str:
    """Lists every customer segment with size, revenue share, median recency/frequency/spend
    and whether it is a churn-risk segment. Takes no input."""
    return json.dumps(list_segments(), default=str, indent=1)


@tool("Get segment brief")
def segment_brief_tool(segment_name: str) -> str:
    """Returns the full behavioural brief for ONE segment, given its exact name (for example
    'At-Risk Whales'): size, revenue share, median days since last order, order counts, spend,
    average order value, return rate, how many customers last ordered in each recency band,
    the top products they buy, top countries, and the incentive policy limits for the segment."""
    try:
        return json.dumps(segment_brief(segment_name.strip().strip("'\"")), default=str, indent=1)
    except ValueError as exc:
        return f"Error: {exc}"


# --------------------------------------------------------------------------------------------
# Crew
# --------------------------------------------------------------------------------------------

def check_ollama(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> None:
    """Fail fast with an actionable message instead of a CrewAI stack trace."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:  # noqa: S310
            tags = json.load(resp)
    except OSError as exc:
        raise AgentSetupError(
            f"Can't reach Ollama at {base_url} ({exc}). Start it with `ollama serve`.") from exc
    pulled = {m["name"] for m in tags.get("models", [])}
    if model not in pulled and f"{model}:latest" not in pulled:
        raise AgentSetupError(
            f"Ollama model {model!r} isn't pulled (available: {sorted(pulled) or 'none'}). "
            f"Run `ollama pull {model}` or set OLLAMA_MODEL to one of the available models.")


def build_llm(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> LLM:
    return LLM(model=f"ollama/{model}", base_url=base_url, temperature=0.4)


def build_crew(segment_name: str, llm: LLM) -> Crew:
    policy = INCENTIVE_POLICY.get(segment_name, DEFAULT_POLICY)
    analyst = Agent(
        role="Lifecycle Marketing Analyst",
        goal="Turn RFM cohort data into retention campaigns that win customers back profitably.",
        backstory=(
            "You run lifecycle marketing for a UK online giftware retailer that sells mostly to "
            "small shops and wholesalers. Customers were segmented with K-Means on recency, "
            "frequency and net spend. You never invent numbers: every figure you use comes from "
            "the segment tools. You respect the discount limits set by finance."),
        tools=[list_segments_tool, segment_brief_tool],
        llm=llm,
        max_iter=6,
        allow_delegation=False,
        verbose=False,
    )

    analysis = Task(
        description=(
            f"Call the 'Get segment brief' tool with segment_name '{segment_name}'. Using only the "
            "returned data, write a short cohort analysis: who these customers are, how valuable "
            "they are (customers, revenue share, median spend and orders), how long since they "
            "last ordered, what they buy (name the top products), and the main retention risk. Quote "
            "durations exactly as given in days (e.g. 180 days is about six months, "
            "not a year); do not round them up or reinterpret them."),
        expected_output=(
            "A markdown section titled '## Cohort snapshot' with 4-6 bullet points of real "
            "figures from the tool, then one sentence naming the main retention risk."),
        agent=analyst,
    )

    campaign = Task(
        description=(
            f"Write a retention email campaign for the '{segment_name}' segment. "
            f"Objective: {policy['objective']}. Preferred incentive: {policy['preferred_incentive']}. "
            f"Any discount must be at most {policy['max_discount_pct']}%. Reference the segment's "
            "actual top products by name. Keep the tone warm and specific to a retailer that "
            "buys giftware for resale. Use the merge fields *|FNAME|* and *|SHOP|* exactly as written "
            "(they are filled in by the email platform)."),
        expected_output=(
            "Markdown with these sections, in order: the '## Cohort snapshot' from the analysis "
            "task; '## Campaign strategy' (objective, incentive and why, when to send relative to "
            "today, success metric); '## Email' containing exactly two subject lines labelled "
            "'Subject A:' and 'Subject B:', a "
            "'Preview text:' line, and a 120-180 word email body ending with a clear call to "
            "action; '## Follow-up' with one sentence on a reminder email for non-openers."),
        agent=analyst,
        context=[analysis],
        markdown=True,
        guardrail=make_campaign_guardrail(segment_name),
        guardrail_max_retries=2,
    )
    return Crew(agents=[analyst], tasks=[analysis, campaign], process=Process.sequential, verbose=False)


# "15% off", "15 % discount", "discount of 15%", "save 15%". Plain percentages elsewhere
# (e.g. "16.4% of revenue" in the snapshot) are data, not offers, and are ignored.
OFFER_PCT = re.compile(
    r"(\d{1,2})\s?%\s*(?:off|discount)|(?:discount|save|saving)\s+(?:of\s+|up to\s+)?(\d{1,2})\s?%",
    re.IGNORECASE)


def discount_violations(text: str, segment_name: str) -> list[int]:
    """Offered discount percentages above the segment's cap (empty list = within policy)."""
    cap = INCENTIVE_POLICY.get(segment_name, DEFAULT_POLICY)["max_discount_pct"]
    offers = {int(a or b) for a, b in OFFER_PCT.findall(text)}
    return sorted(p for p in offers if p > cap)


REQUIRED_SECTIONS = ["## Cohort snapshot", "## Campaign strategy", "## Email", "## Follow-up"]


def campaign_problems(text: str, segment_name: str) -> list[str]:
    """Format and policy problems in a campaign draft (empty list = acceptable)."""
    lower = text.lower()
    problems = [f"missing the '{h}' section" for h in REQUIRED_SECTIONS if h.lower() not in lower]
    if "subject a:" not in lower or "subject b:" not in lower:
        problems.append("needs exactly two subject lines labelled 'Subject A:' and 'Subject B:'")
    if "*|fname|*" not in lower:
        problems.append("the email must use the merge field *|FNAME|*")
    cap = INCENTIVE_POLICY.get(segment_name, DEFAULT_POLICY)["max_discount_pct"]
    if over := discount_violations(text, segment_name):
        problems.append(f"offers {over}% off but this segment's cap is {cap}%")
    return problems


def make_campaign_guardrail(segment_name: str):
    """CrewAI task guardrail: on failure the agent gets the problems back and retries."""
    # Returns (ok, result_or_feedback). Deliberately unannotated: CrewAI inspects the return
    # annotation at runtime, and `from __future__ import annotations` turns it into a string
    # that its check rejects.
    def guardrail(output):
        problems = campaign_problems(output.raw, segment_name)
        if problems:
            return False, ("Rewrite the full campaign. Fix these problems: " + "; ".join(problems)
                           + ". Use exactly the markdown section headings requested.")
        return True, output.raw
    return guardrail


def generate_retention_campaign(segment_name: str, save: bool = True) -> str:
    """Run the crew for one segment and return the campaign markdown (also saved to campaigns/)."""
    check_ollama()
    segment_brief(segment_name)  # validates the name and the marts before spending LLM time
    result = build_crew(segment_name, build_llm()).kickoff()
    text = str(result).strip()
    problems = campaign_problems(text, segment_name)
    status = ("Passed format and discount-policy checks" if not problems
              else "**Failed checks after retries: " + "; ".join(problems) + "**")
    header = (f"# Retention campaign: {segment_name}\n\n"
              f"_Generated {datetime.now():%Y-%m-%d %H:%M} by a local CrewAI agent "
              f"(Ollama `{OLLAMA_MODEL}`). {status}. Review before sending._\n\n")
    text = header + text
    if save:
        CAMPAIGN_DIR.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", segment_name.lower()).strip("_")
        path = CAMPAIGN_DIR / f"{slug}_{datetime.now():%Y%m%d_%H%M%S}.md"
        path.write_text(text + "\n")
        log.info("Saved %s", path.relative_to(REPO_ROOT))
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--segment", help="exact segment name, e.g. 'At-Risk Whales'")
    group.add_argument("--all-risk", action="store_true", help="every churn-risk segment")
    args = parser.parse_args(argv)
    # CrewAI and httpx log every request at INFO on the root logger; keep only our own progress.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)

    try:
        segments = ([s["segment_name"] for s in list_segments() if s["is_churn_risk"]]
                    if args.all_risk else [args.segment])
        for seg in segments:
            print(generate_retention_campaign(seg), end="\n\n")
    except (AgentSetupError, ValueError, duckdb.Error) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
