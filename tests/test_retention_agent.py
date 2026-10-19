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


GOOD_CAMPAIGN = """## Cohort snapshot
- 1,950 customers
## Campaign strategy
One-time 20% off with a 14-day expiry.
## Email
Subject A: We miss you, *|FNAME|*
Subject B: A little something for *|SHOP|*
Preview text: 20% off for 14 days
Dear *|FNAME|*, ...
## Follow-up
Remind non-openers after 5 days.
"""


def test_good_campaign_has_no_problems():
    assert ra.campaign_problems(GOOD_CAMPAIGN, "Dormant Buyers") == []


def test_campaign_problems_lists_every_issue():
    bad = "**Campaign Objective:** reactivate\nSpecial discount of 30% off!"
    problems = ra.campaign_problems(bad, "Dormant Buyers")
    assert any("Cohort snapshot" in p for p in problems)
    assert any("Subject A" in p for p in problems)
    assert any("FNAME" in p for p in problems)
    assert any("[30]% off" in p and "cap is 20%" in p for p in problems)


def test_guardrail_returns_feedback_for_retry():
    class Out:
        raw = GOOD_CAMPAIGN.replace("Subject B: A little something for *|SHOP|*\n", "")
    ok, feedback = ra.make_campaign_guardrail("Dormant Buyers")(Out())
    assert ok is False and "Subject B" in feedback
    Out.raw = GOOD_CAMPAIGN
    assert ra.make_campaign_guardrail("Dormant Buyers")(Out()) == (True, GOOD_CAMPAIGN)


def test_crew_wiring_is_valid_without_calling_the_llm():
    # Constructing the crew runs CrewAI's pydantic validation (tools, guardrail signature,
    # task context) without any network call, so wiring mistakes fail here, not mid-demo.
    crew = ra.build_crew("At-Risk Whales", ra.build_llm())
    analysis, campaign = crew.tasks
    assert campaign.context == [analysis]
    assert campaign.guardrail is not None
    assert {t.name for t in crew.agents[0].tools} == {"List customer segments", "Get segment brief"}
