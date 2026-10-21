"""Tests for agent/retention_agent.py that don't need an LLM.

The data tools run against the real retail_analytics.duckdb (skipped if it hasn't been built);
the Ollama check and the discount guardrail are tested with fakes.
"""

import io
import json

import pytest

import agent.retention_agent as ra

needs_db = pytest.mark.skipif(not ra.DB_PATH.exists(), reason="retail_analytics.duckdb not built")


@needs_db
def test_list_segments_returns_the_four_personas():
    names = [s["segment_name"] for s in ra.list_segments()]
    assert sorted(names) == sorted(ra.INCENTIVE_POLICY)


@needs_db
def test_segment_brief_is_complete_and_json_serialisable():
    brief = ra.segment_brief("Dormant Buyers")
    json.dumps(brief)
    assert brief["customers"] > 0 and brief["is_churn_risk"] is True
    assert len(brief["top_products"]) == 5
    assert brief["incentive_policy"]["max_discount_pct"] == 20
    bands = ["ordered_last_90d", "last_order_91_180d", "last_order_181_365d", "last_order_over_1y"]
    assert sum(brief[b] for b in bands) == brief["customers"]


@needs_db
def test_unknown_segment_raises_with_valid_names():
    with pytest.raises(ValueError, match="Loyal Champions"):
        ra.segment_brief("Big Spenders")


@needs_db
def test_tool_returns_error_text_instead_of_raising():
    out = ra.segment_brief_tool.run(segment_name="Nope")
    assert out.startswith("Error:")


@needs_db
def test_tool_tolerates_quoted_names():
    out = ra.segment_brief_tool.run(segment_name="'At-Risk Whales'")
    assert json.loads(out)["segment_name"] == "At-Risk Whales"


@pytest.mark.parametrize("text, expected", [
    ("Enjoy 15% off your next order", []),
    ("Enjoy 25% off your next order", [25]),
    ("a discount of 30% today", [30]),
    ("save up to 40%!", [40]),
    ("Take 100% off a sample pack", [100]),
    ("25 percent off this week", [25]),
    ("This segment is 16.4% of revenue and 24% of customers", []),  # data, not offers
])
def test_discount_guardrail_for_at_risk_whales(text, expected):
    assert ra.discount_violations(text, "At-Risk Whales") == expected


def fake_urlopen(payload):
    def _open(url, timeout=None):
        return io.BytesIO(json.dumps(payload).encode())
    return _open


def test_check_ollama_accepts_latest_tag(monkeypatch):
    monkeypatch.setattr(ra.urllib.request, "urlopen", fake_urlopen({"models": [{"name": "llama3.1:latest"}]}))
    ra.check_ollama("llama3.1")


def test_check_ollama_names_the_pull_command(monkeypatch):
    monkeypatch.setattr(ra.urllib.request, "urlopen", fake_urlopen({"models": [{"name": "mistral:latest"}]}))
    with pytest.raises(ra.AgentSetupError, match="ollama pull llama3.1"):
        ra.check_ollama("llama3.1")


def test_check_ollama_explains_when_server_is_down(monkeypatch):
    def refuse(url, timeout=None):
        raise ConnectionRefusedError("refused")
    monkeypatch.setattr(ra.urllib.request, "urlopen", refuse)
    with pytest.raises(ra.AgentSetupError, match="ollama serve"):
        ra.check_ollama("llama3.1")


GOOD_CAMPAIGN = """## Campaign strategy
One-time 20% off with a 14-day expiry.
## Email
Subject A: We miss you, *|FNAME|*
Subject B: A little something for *|SHOP|*
Preview text: 20% off for 14 days
Dear *|FNAME|*, it has been a while since your last order, and a lot has changed in our catalogue.
Our cake stands, retrospot baking sets and hanging heart T-light holders are back in stock in new
colours, and they still sell well in independent gift shops ahead of the busy season. To make
restocking easy, here is 20% off your next order for the next 14 days, with no minimum spend and
no strings attached. Everything you need for your shelves is one click away, and our team is
happy to help you choose bestsellers for *|SHOP|* if you reply to this email with your list.
We would love to have you back. Shop the new ranges today before the code expires.
## Follow-up
Remind non-openers after 5 days.
"""


def test_good_campaign_has_no_problems():
    assert ra.campaign_problems(GOOD_CAMPAIGN, "Dormant Buyers") == []


def test_campaign_problems_lists_every_issue():
    bad = "**Campaign Objective:** reactivate\nSpecial discount of 30% off!"
    problems = ra.campaign_problems(bad, "Dormant Buyers")
    assert any("Campaign strategy" in p for p in problems)
    assert any("Subject A" in p for p in problems)
    assert any("FNAME" in p for p in problems)
    assert any("[30]% off" in p and "cap is 20%" in p for p in problems)


def test_guardrail_returns_feedback_then_accepts_after_retries():
    class Out:
        raw = GOOD_CAMPAIGN.replace("Subject B: A little something for *|SHOP|*\n", "")
    guard = ra.make_campaign_guardrail("Dormant Buyers", retries=2)
    for _ in range(2):
        ok, feedback = guard(Out())
        assert ok is False and "Subject B" in feedback
    # Retries exhausted: the draft is accepted so the user gets something to edit...
    assert guard(Out()) == (True, Out.raw)
    # ...and the assembled document flags it.
    doc = ra.assemble_campaign("Dormant Buyers", FAKE_BRIEF, "- notes", Out.raw, "llama3.2")
    assert "Failed checks after retries" in doc and "Subject B" in doc


def test_good_draft_passes_guardrail_first_time():
    class Out:
        raw = GOOD_CAMPAIGN
    assert ra.make_campaign_guardrail("Dormant Buyers")(Out()) == (True, GOOD_CAMPAIGN)


@pytest.mark.parametrize("heading", ["## Email", "# Email", "### email", "**Email**", "**Email:**"])
def test_headings_are_matched_at_any_level(heading):
    assert ra.has_heading(f"intro\n{heading}\nbody", "Email")


def test_heading_match_ignores_inline_mentions():
    assert not ra.has_heading("We will send an Email next week.", "Email")


FAKE_BRIEF = {
    "customers": 1950, "share_of_all_customers_pct": 33.4, "share_of_total_net_revenue_pct": 3.7,
    "total_net_spend_gbp": 608954.0, "median_orders": 1.0, "median_net_spend_gbp": 267.0,
    "avg_order_value": 250.0, "median_days_since_last_order": 402.0, "ordered_last_90d": 0.0,
    "last_order_91_180d": 150.0, "last_order_181_365d": 600.0, "last_order_over_1y": 1200.0,
    "avg_return_rate_pct": 1.5,
    "top_products": [{"product": "REGENCY CAKESTAND 3 TIER", "customers_who_bought": 120, "revenue": 1.0}],
    "top_countries": [{"country": "United Kingdom", "customers": 1800}],
}


def test_assembled_campaign_uses_data_snapshot_and_passes():
    doc = ra.assemble_campaign("Dormant Buyers", FAKE_BRIEF, "- They bought once.", GOOD_CAMPAIGN, "llama3.2")
    assert doc.startswith("# Retention campaign: Dormant Buyers")
    assert "**1,950 customers**: 33.4% of active customers, 3.7% of net revenue" in doc
    assert "62% over a year ago" in doc and "Regency Cakestand 3 Tier (120 buyers)" in doc
    assert "**Typical customer**: 1 order, £267" in doc
    assert "## Analyst notes\n\n- They bought once." in doc
    assert "Passed format and discount-policy checks" in doc


def test_email_without_body_is_rejected():
    no_body = GOOD_CAMPAIGN.split("Dear *|FNAME|*")[0] + "## Follow-up\nRemind them.\n"
    problems = ra.campaign_problems(no_body, "Dormant Buyers")
    assert any("email body has only" in p for p in problems)


def test_clean_removes_echoed_tool_calls_and_unwraps_fences():
    raw = ('notes\n```json\n{"name": "get_segment_brief", "parameters": {"segment_name": "x"}}\n```\n'
           "## Email\n```markdown\nSubject A: hi\n```\n")
    out = ra.clean_llm_markdown(raw)
    assert "get_segment_brief" not in out and "```" not in out and "Subject A: hi" in out


def test_crew_wiring_is_valid_without_calling_the_llm():
    # Constructing the crew runs CrewAI's pydantic validation (tools, guardrail signature,
    # task context) without any network call, so wiring mistakes fail here, not mid-demo.
    crew = ra.build_crew("At-Risk Whales", ra.build_llm())
    analysis, campaign = crew.tasks
    assert campaign.context == [analysis]
    assert campaign.guardrail is not None
    assert {t.name for t in crew.agents[0].tools} == {"List customer segments", "Get segment brief"}
