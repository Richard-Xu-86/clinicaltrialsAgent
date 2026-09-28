"""End-to-end over recorded real API responses (examples/recordings).

The key property checked here is the citation guarantee: every evidence value
in every citation must literally exist at the stated path in the raw API record.
"""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ctviz import api
from ctviz.pipeline import VisualizationService
from ctviz.schemas import VisualizeRequest
from tests.conftest import RECORDINGS

EXAMPLES = json.loads((Path(__file__).resolve().parents[1] / "examples" / "requests.json").read_text())


def raw_studies_by_id() -> dict[str, list[dict]]:
    """NCT ID -> every recorded copy (the same trial appears in several recordings,
    each fetched with a different field list)."""
    studies: dict[str, list[dict]] = {}
    for path in RECORDINGS.glob("*.json"):
        body = json.loads(path.read_text())["body"]
        for s in body.get("studies", []):
            studies.setdefault(s["protocolSection"]["identificationModule"]["nctId"], []).append(s)
    return studies


def resolve_path(obj, path: str):
    for part in path.split("."):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        obj = obj[m.group(1)]
        if m.group(2) is not None:
            obj = obj[int(m.group(2))]
    return ", ".join(obj) if isinstance(obj, list) else str(obj)


def value_in_any_copy(copies: list[dict], path: str):
    for study in copies:
        try:
            return resolve_path(study, path)
        except (KeyError, IndexError, TypeError):
            continue
    raise AssertionError(f"path {path} not found in any recorded copy")


def field_present_anywhere(copies: list[dict], path: str) -> bool:
    try:
        value_in_any_copy(copies, path)
        return True
    except AssertionError:
        return False


def iter_data(viz: dict):
    if viz["type"] == "network_graph":
        yield from viz["data"]["nodes"]
        yield from viz["data"]["edges"]
    else:
        yield from viz["data"]


@pytest.fixture(scope="module")
def raw():
    return raw_studies_by_id()


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["name"] for e in EXAMPLES])
def test_example_citations_are_exact(example, offline_settings, raw):
    request = VisualizeRequest(**{**example["request"], "max_citations_per_datum": 1000})
    response = VisualizationService(offline_settings).visualize(request).model_dump(mode="json")
    viz = response["visualization"]

    checked = 0
    for datum in iter_data(viz):
        count = datum.get("trial_count", datum.get("citation_count"))
        # With a high limit, every counted trial is cited exactly once per datum.
        assert len(datum["citations"]) == datum["citation_count"] == count
        assert len({c["nct_id"] for c in datum["citations"]}) == len(datum["citations"])
        for citation in datum["citations"]:
            copies = raw[citation["nct_id"]]
            assert citation["evidence"], f"{citation['nct_id']} cited without evidence"
            for ev in citation["evidence"]:
                if ev["value"] is None:  # "field absent" evidence, e.g. the "No phase" bucket
                    assert not field_present_anywhere(copies, ev["field"]), (citation["nct_id"], ev)
                else:
                    assert value_in_any_copy(copies, ev["field"]) == ev["value"], (citation["nct_id"], ev)
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["name"] for e in EXAMPLES])
def test_example_counts_are_consistent(example, offline_settings):
    response = VisualizationService(offline_settings).visualize(VisualizeRequest(**example["request"]))
    for cohort in response.meta.cohorts:
        assert cohort.analyzed + cohort.excluded == cohort.fetched <= cohort.api_total
    viz = response.visualization
    # Phase is single-valued (multi-phase trials get a combined bucket), so a phase bar
    # chart must account for every analyzed trial exactly once.
    if viz.type.value == "bar_chart" and viz.encoding.x.field == "phase":
        assert sum(r["trial_count"] for r in viz.data) == response.meta.cohorts[0].analyzed


# --------------------------------------------------------------------------- HTTP layer


@pytest.fixture
def http(offline_settings):
    api.app.dependency_overrides[api.get_service] = lambda: VisualizationService(offline_settings)
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_http_ok(http):
    body = {"query": "How are Huntington's disease trials distributed across phases?"}
    res = http.post("/v1/visualize", json=body)
    assert res.status_code == 200
    payload = res.json()
    assert payload["visualization"]["type"] == "bar_chart"
    assert payload["meta"]["planner"]["method"] == "rule_based"


def test_http_validation_error(http):
    assert http.post("/v1/visualize", json={"query": "x" * 3, "trial_phase": "Phase 9"}).status_code == 422


def test_http_offline_miss_is_503(http):
    res = http.post("/v1/visualize", json={"query": "How are gout trials distributed across phases?"})
    assert res.status_code == 503


def test_http_upstream_error_is_502(fast_settings, monkeypatch):
    from ctviz.ctgov.client import CTGovClient, CTGovError

    def boom(self, params, max_records):
        raise CTGovError("down")

    monkeypatch.setattr(CTGovClient, "search", boom)
    api.app.dependency_overrides[api.get_service] = lambda: VisualizationService(fast_settings)
    try:
        res = TestClient(api.app).post("/v1/visualize", json={"query": "melanoma trials by phase"})
        assert res.status_code == 502
    finally:
        api.app.dependency_overrides.clear()


def test_schema_endpoint(http):
    schemas = http.get("/v1/schema").json()
    assert set(schemas) == {"request", "plan", "response"}

