"""Request validation, rule-based planning, plan resolution and the LLM planner loop."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ctviz.ctgov.query import build_search_params
from ctviz.planner import plan_request
from ctviz.planner.llm import OpenAIPlanner, PlannerError, plan_tool_schema
from ctviz.planner.resolve import resolve_plan
from ctviz.planner.rules import RuleBasedPlanner
from ctviz.schemas import (
    AnalysisType,
    CompareSpec,
    Dimension,
    EntityType,
    OverallStatus,
    Phase,
    PlanFilters,
    QueryPlan,
    StudyType,
    VisualizeRequest,
)

# --------------------------------------------------------------------------- request


class TestRequest:
    def test_minimal(self):
        assert VisualizeRequest(query="phases of melanoma trials").max_records == 3000

    @pytest.mark.parametrize("value", ["Phase 3", "PHASE3", "3", "phase iii", "Phase-3"])
    def test_phase_spellings(self, value):
        assert VisualizeRequest(query="xyz", trial_phase=value).trial_phase == [Phase.PHASE3]

    def test_status_spelling(self):
        req = VisualizeRequest(query="xyz", status=["recruiting", "not yet recruiting"])
        assert req.status == [OverallStatus.RECRUITING, OverallStatus.NOT_YET_RECRUITING]

    @pytest.mark.parametrize("bad", [
        {"trial_phase": "Phase 7"}, {"start_year": 2020, "end_year": 2010}, {"start_year": 1850},
        {"unknown_field": 1}, {"query": ""}, {"max_records": 0},
    ])
    def test_rejections(self, bad):
        with pytest.raises(ValidationError):
            VisualizeRequest(**{"query": "some question", **bad})


# --------------------------------------------------------------------------- rule planner

RULE_CASES = [
    ("How has the number of trials for pembrolizumab changed per year since 2015?",
     dict(analysis="trend", drug="pembrolizumab", start_year=2015)),
    ("How many trials started each year for Alzheimer's disease?",
     dict(analysis="trend", condition="Alzheimer's disease", series_by=None)),
    ("How are glioblastoma trials distributed across phases?",
     dict(analysis="distribution", condition="glioblastoma", group_by="phase")),
    ("What are the most common intervention types for breast cancer trials?",
     dict(analysis="distribution", condition="breast cancer", group_by="intervention_type")),
    ("Compare phases for trials involving semaglutide vs tirzepatide.",
     dict(analysis="comparison", group_by="phase", compare=["semaglutide", "tirzepatide"])),
    ("Compare sponsor categories across lung cancer and melanoma.",
     dict(analysis="comparison", group_by="sponsor_class", compare=["lung cancer", "melanoma"])),
    ("Which countries have the most recruiting trials for Duchenne muscular dystrophy?",
     dict(analysis="distribution", group_by="country", statuses=["RECRUITING"],
          condition="Duchenne muscular dystrophy")),
    ("Show a network of sponsors and drugs for recruiting glioblastoma trials.",
     dict(analysis="network", network=("sponsor", "drug"), condition="glioblastoma")),
    ("Which drugs frequently co-occur in combination studies for melanoma?",
     dict(analysis="network", network=("drug", "drug"), condition="melanoma")),
    ("What is the distribution of enrollment sizes for Phase 3 Alzheimer's trials?",
     dict(analysis="histogram", phases=["PHASE3"])),
    ("Plot enrollment against duration for completed obesity trials",
     dict(analysis="scatter", statuses=["COMPLETED"], condition="obesity")),
]


@pytest.mark.parametrize("query,expected", RULE_CASES)
def test_rule_planner(query, expected):
    plan = RuleBasedPlanner().plan(query)
    assert plan.analysis.value == expected["analysis"]
    f = plan.filters
    for key in ("drug", "condition", "start_year"):
        if key in expected:
            assert getattr(f, key) == expected[key]
    if "phases" in expected:
        assert [p.value for p in f.phases] == expected["phases"]
    if "statuses" in expected:
        assert [s.value for s in f.statuses] == expected["statuses"]
    if "group_by" in expected:
        assert plan.group_by.value == expected["group_by"]
    if "series_by" in expected:
        assert plan.series_by == expected["series_by"]
    if "compare" in expected:
        assert plan.compare.values == expected["compare"]
    if "network" in expected:
        assert (plan.network.source.value, plan.network.target.value) == expected["network"]


# --------------------------------------------------------------------------- resolution


def _resolve(plan: QueryPlan, **req):
    return resolve_plan(plan, VisualizeRequest(query="question text", **req))


class TestResolve:
    def test_explicit_field_overrides_plan(self):
        plan = QueryPlan(analysis="trend", filters=PlanFilters(drug="nivolumab"))
        res = _resolve(plan, drug_name="Pembrolizumab")
        assert res.plan.filters.drug == "Pembrolizumab"
        assert res.overrides and "nivolumab" in res.overrides[0]

    def test_explicit_drug_joins_comparison_instead_of_filtering(self):
        plan = QueryPlan(analysis="comparison", group_by="phase",
                         compare=CompareSpec(field="drug", values=["nivolumab", "atezolizumab"]))
        res = _resolve(plan, drug_name="pembrolizumab")
        assert res.plan.filters.drug is None
        assert res.plan.compare.values[0] == "pembrolizumab"

    def test_single_value_comparison_becomes_distribution(self):
        plan = QueryPlan(analysis="comparison", group_by="phase",
                         compare=CompareSpec(field="drug", values=["x", "X"]))
        res = _resolve(plan)
        assert res.plan.analysis == AnalysisType.DISTRIBUTION
        assert res.plan.filters.drug == "x" and res.adjustments

    def test_trend_moves_grouping_to_series(self):
        res = _resolve(QueryPlan(analysis="trend", group_by="phase"))
        assert res.plan.group_by == Dimension.START_YEAR and res.plan.series_by == Dimension.PHASE

    def test_distribution_by_year_becomes_trend(self):
        assert _resolve(QueryPlan(analysis="distribution", group_by="start_year")).plan.analysis == "trend"

    def test_comparison_cannot_group_by_its_own_field(self):
        plan = QueryPlan(analysis="comparison", group_by="drug",
                         compare=CompareSpec(field="drug", values=["a", "b"]))
        assert _resolve(plan).plan.group_by == Dimension.PHASE

    def test_network_defaults(self):
        res = _resolve(QueryPlan(analysis="network", filters=PlanFilters(condition="melanoma")))
        net = res.plan.network
        assert (net.source, net.target, net.min_edge_weight) == (EntityType.SPONSOR, EntityType.DRUG, 1)
        res = _resolve(QueryPlan(analysis="network"))
        assert res.plan.network.source == res.plan.network.target == EntityType.DRUG
        assert res.plan.network.min_edge_weight == 2

    def test_scatter_needs_two_fields(self):
        plan = QueryPlan(analysis="scatter", numeric={"x": "enrollment", "y": "enrollment"})
        assert _resolve(plan).plan.numeric.y.value == "duration_months"

    def test_irrelevant_parts_cleared(self):
        plan = QueryPlan(analysis="histogram", group_by="phase", network={"source": "drug", "target": "drug"})
        resolved = _resolve(plan).plan
        assert resolved.group_by is None and resolved.network is None and resolved.numeric is not None


# --------------------------------------------------------------------------- API params


def test_search_params():
    filters = PlanFilters(drug="pembrolizumab", condition="melanoma", sponsor="Merck",
                          country="United States", phases=[Phase.PHASE2, Phase.PHASE3],
                          statuses=[OverallStatus.RECRUITING], study_type=StudyType.INTERVENTIONAL,
                          start_year=2015)
    params = build_search_params(filters, ["Phase", "NCTId"])
    assert params["query.intr"] == "pembrolizumab"
    assert params["query.lead"] == "Merck"  # lead sponsor only, not collaborators
    assert params["filter.overallStatus"] == "RECRUITING"
    assert params["filter.advanced"] == (
        'AREA[Phase](PHASE2 OR PHASE3) AND AREA[LocationCountry]"United States" AND '
        "AREA[StudyType]INTERVENTIONAL AND AREA[StartDate]RANGE[2015-01-01,MAX]"
    )
    assert params["fields"] == "NCTId,Phase"


# --------------------------------------------------------------------------- LLM planner (OpenAI)


class FakeChatCompletions:
    """Stands in for openai.OpenAI().chat.completions; arguments arrive as JSON strings,
    or an Exception to raise, or None for a reply without a tool call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        tool_calls = [] if reply is None else [SimpleNamespace(
            id=f"call_{len(self.calls)}", function=SimpleNamespace(name="submit_query_plan", arguments=reply))]
        message = SimpleNamespace(content=None, tool_calls=tool_calls, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _llm(replies, settings):
    fake = FakeChatCompletions(replies)
    return OpenAIPlanner(settings, client=SimpleNamespace(chat=SimpleNamespace(completions=fake))), fake


class TestLLMPlanner:
    def test_tool_schema_is_self_contained(self):
        assert "$ref" not in str(plan_tool_schema()) and "$defs" not in plan_tool_schema()

    def test_valid_plan(self, fast_settings):
        planner, fake = _llm(['{"analysis": "distribution", "group_by": "phase", '
                              '"filters": {"condition": "melanoma"}}'], fast_settings)
        plan = planner.plan(VisualizeRequest(query="melanoma phases", condition="melanoma"))
        assert plan.group_by == Dimension.PHASE
        call = fake.calls[0]
        assert call["tool_choice"] == {"type": "function", "function": {"name": "submit_query_plan"}}
        assert call["messages"][0]["role"] == "system" and "temperature" not in call
        assert "melanoma" in call["messages"][1]["content"]  # explicit fields are passed on

    def test_invalid_plan_is_sent_back_and_corrected(self, fast_settings):
        planner, fake = _llm(['{"analysis": "pie_chart"}', '{"analysis": "trend"}'], fast_settings)
        assert planner.plan(VisualizeRequest(query="trend please")).analysis == AnalysisType.TREND
        retry = fake.calls[1]["messages"]
        assert retry[-1]["role"] == "tool" and retry[-1]["tool_call_id"] == "call_1"
        assert retry[-2]["tool_calls"][0]["id"] == "call_1"  # assistant turn echoed before the tool result

    def test_bad_json_then_invalid_gives_up(self, fast_settings):
        planner, _ = _llm(['{"analysis": ', '{"analysis": "pie"}'], fast_settings)
        with pytest.raises(PlannerError):
            planner.plan(VisualizeRequest(query="whatever"))

    def test_missing_tool_call_is_nudged(self, fast_settings):
        planner, fake = _llm([None, '{"analysis": "trend"}'], fast_settings)
        assert planner.plan(VisualizeRequest(query="trend please")).analysis == AnalysisType.TREND
        assert fake.calls[1]["messages"][-1]["role"] == "user"

    def test_failure_falls_back_to_rules(self, fast_settings):
        planner, _ = _llm([RuntimeError("rate limited")], fast_settings)
        info = plan_request(VisualizeRequest(query="How are glioblastoma trials distributed across phases?"),
                            fast_settings, llm=planner)
        assert info.method == "rule_based" and "rate limited" in info.fallback_reason
        assert info.plan.filters.condition == "glioblastoma"

    def test_missing_sdk_falls_back(self, fast_settings, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        settings = fast_settings.model_copy(update={"openai_api_key": "sk-test"})
        info = plan_request(VisualizeRequest(query="phases of melanoma trials"), settings)
        assert info.method == "rule_based" and "not installed" in info.fallback_reason

    def test_no_key_uses_rules(self, fast_settings):
        info = plan_request(VisualizeRequest(query="phases of melanoma trials"), fast_settings)
        assert info.method == "rule_based" and info.fallback_reason == "OPENAI_API_KEY is not set"

    def test_model_name_reported(self, fast_settings):
        planner, _ = _llm(['{"analysis": "distribution", "group_by": "phase"}'], fast_settings)
        info = plan_request(VisualizeRequest(query="phases please"), fast_settings, llm=planner)
        assert info.method == "llm" and info.model == "gpt-5.4-mini"


# --------------------------------------------------------------------------- regressions from review


@pytest.mark.parametrize("query,drug,values", [
    ("pembrolizumab vs nivolumab vs atezolizumab phases", None, ["pembrolizumab", "nivolumab", "atezolizumab"]),
    ("Compare this drug with nivolumab by phase", "pembrolizumab", ["pembrolizumab", "nivolumab"]),
    ("Compare trials of Keytruda and Opdivo by phase", None, ["Keytruda", "Opdivo"]),
    ("Compare recruiting trials in breast cancer, lung cancer and melanoma by phase", None,
     ["breast cancer", "lung cancer", "melanoma"]),
])
def test_rule_planner_multiway_comparisons(query, drug, values):
    plan = RuleBasedPlanner().plan(query, explicit_drug=drug)
    assert plan.analysis == AnalysisType.COMPARISON and plan.compare.values == values


def test_phase_range_becomes_both_phases():
    plan = RuleBasedPlanner().plan("How many phase 1/2 melanoma trials by status?")
    assert plan.filters.phases == [Phase.PHASE1, Phase.PHASE2]


def test_explicit_start_year_is_never_moved():
    plan = QueryPlan(analysis="trend", filters=PlanFilters(end_year=2010))
    res = _resolve(plan, start_year=2015)
    assert (res.plan.filters.start_year, res.plan.filters.end_year) == (2015, None)
    assert res.adjustments


def test_explicit_drug_replaces_last_cohort_when_full():
    plan = QueryPlan(analysis="comparison", group_by="phase",
                     compare=CompareSpec(field="drug", values=["a", "b", "c", "d", "e"]))
    res = _resolve(plan, drug_name="pembrolizumab")
    assert res.plan.compare.values == ["pembrolizumab", "a", "b", "c", "d"]
    assert any("dropped 'e'" in o for o in res.overrides)
