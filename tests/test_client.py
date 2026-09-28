"""CTGovClient: pagination, retries, caching and offline replay (HTTP mocked with respx)."""

import httpx
import pytest
import respx

from ctviz.ctgov import client as client_module
from ctviz.ctgov.client import CTGovClient, CTGovError, OfflineCacheMiss, build_url

BASE = "https://clinicaltrials.gov/api/v2"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda s: None)


def study(nct):
    return {"protocolSection": {"identificationModule": {"nctId": nct}}}


def test_build_url_is_deterministic():
    a = build_url(BASE, "studies", {"b": "2", "a": "1", "empty": None})
    assert a == build_url(BASE, "studies", {"a": "1", "b": "2"}) == f"{BASE}/studies?a=1&b=2"


@respx.mock
def test_paginates_until_cap(fast_settings):
    route = respx.get(f"{BASE}/studies").mock(side_effect=[
        httpx.Response(200, json={"totalCount": 2500, "studies": [study(f"NCT{i}") for i in range(1000)],
                                  "nextPageToken": "t1"}),
        httpx.Response(200, json={"studies": [study(f"NCT{i}") for i in range(1000, 2000)],
                                  "nextPageToken": "t2"}),
    ])
    result = CTGovClient(fast_settings).search({"query.cond": "x"}, max_records=1500)
    assert route.call_count == 2
    assert "pageToken=t1" in str(route.calls[1].request.url)
    assert len(result.studies) == 1500 and result.total_count == 2500 and result.truncated


@respx.mock
def test_retries_transient_errors(fast_settings):
    route = respx.get(f"{BASE}/studies").mock(side_effect=[
        httpx.Response(503), httpx.Response(429, headers={"Retry-After": "1"}),
        httpx.Response(200, json={"totalCount": 1, "studies": [study("NCT1")]}),
    ])
    result = CTGovClient(fast_settings).search({}, max_records=10)
    assert route.call_count == 3 and not result.truncated


@respx.mock
def test_client_errors_are_not_retried(fast_settings):
    route = respx.get(f"{BASE}/studies").mock(return_value=httpx.Response(400, text="bad field"))
    with pytest.raises(CTGovError, match="400"):
        CTGovClient(fast_settings).search({}, max_records=10)
    assert route.call_count == 1


@respx.mock
def test_gives_up_after_retries(fast_settings):
    respx.get(f"{BASE}/studies").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(CTGovError, match="unavailable"):
        CTGovClient(fast_settings).search({}, max_records=10)


@respx.mock
def test_cache_hit_avoids_second_request(fast_settings):
    route = respx.get(f"{BASE}/studies").mock(
        return_value=httpx.Response(200, json={"totalCount": 1, "studies": [study("NCT1")]}))
    client = CTGovClient(fast_settings)
    client.search({"query.cond": "x"}, 10)
    client.search({"query.cond": "x"}, 10)
    assert route.call_count == 1


def test_offline_miss_raises(fast_settings):
    settings = fast_settings.model_copy(update={"ctgov_offline": True})
    with pytest.raises(OfflineCacheMiss):
        CTGovClient(settings).search({"query.cond": "never recorded"}, 10)
