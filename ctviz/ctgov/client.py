"""Thin client for the ClinicalTrials.gov v2 REST API.

Responsibilities, and nothing more:
  * build deterministic URLs (so responses can be cached and replayed),
  * paginate with ``pageToken`` up to a record cap,
  * stay under the published rate limit and retry transient failures.

It returns raw study JSON. Interpreting that JSON is the normalizer's job.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from ..config import Settings
from .cache import ResponseCache

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class CTGovError(RuntimeError):
    """The API could not be reached or rejected the request."""


class OfflineCacheMiss(CTGovError):
    """Offline mode is on and the needed response was never recorded."""


@dataclass
class SearchResult:
    studies: list[dict[str, Any]]
    total_count: int
    truncated: bool
    first_page_url: str


class _RateLimiter:
    """Process-wide minimum spacing between outgoing requests."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_limiter: _RateLimiter | None = None


def _get_limiter(interval: float) -> _RateLimiter:
    global _limiter
    if _limiter is None or _limiter.min_interval_s != interval:
        _limiter = _RateLimiter(interval)
    return _limiter


def build_url(base_url: str, path: str, params: dict[str, Any]) -> str:
    """Deterministic URL: parameters sorted, empty values dropped."""
    clean = {k: v for k, v in params.items() if v not in (None, "", [])}
    query = urlencode(sorted(clean.items()))
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{query}" if query else "")


class CTGovClient:
    def __init__(
        self,
        settings: Settings,
        http: httpx.Client | None = None,
        cache: ResponseCache | None = None,
    ):
        self.settings = settings
        self.http = http or httpx.Client(
            timeout=settings.ctgov_timeout_s,
            headers={"User-Agent": "ctviz-agent/1.0 (clinical-trials visualization service)"},
        )
        self.cache = cache or ResponseCache(
            settings.cache_dir, ttl_s=None if settings.ctgov_offline else settings.cache_ttl_s
        )
        self._limiter = _get_limiter(settings.ctgov_min_interval_s)

    # ------------------------------------------------------------------ public API

    def version(self) -> dict[str, Any]:
        """API version and data timestamp. Used for provenance in response metadata."""
        return self._get_json(build_url(self.settings.ctgov_base_url, "version", {}))

    def search(self, params: dict[str, Any], max_records: int) -> SearchResult:
        """Fetch up to ``max_records`` studies matching ``params`` (query./filter./fields)."""
        # Keep the page size fixed for normal requests so cache keys (and recordings) are stable;
        # only small caps shrink it.
        page_size = min(self.settings.ctgov_page_size, max_records)
        base = {**params, "pageSize": page_size, "countTotal": "true", "format": "json"}

        studies: list[dict[str, Any]] = []
        first_url: str | None = None
        total: int | None = None
        token: str | None = None

        while True:
            url = build_url(self.settings.ctgov_base_url, "studies", {**base, "pageToken": token})
            first_url = first_url or url
            body = self._get_json(url)

            if total is None:
                # countTotal is honoured on the first page only.
                total = int(body.get("totalCount", 0))
            studies.extend(body.get("studies", []))
            token = body.get("nextPageToken")

            if not token or len(studies) >= max_records:
                break

        kept = studies[:max_records]
        return SearchResult(
            studies=kept,
            total_count=total or 0,
            truncated=(total or 0) > len(kept),
            first_page_url=first_url,
        )

    # ------------------------------------------------------------------ internals

    def _get_json(self, url: str) -> Any:
        cached = self.cache.get(url)
        if cached is not None:
            return cached
        if self.settings.ctgov_offline:
            raise OfflineCacheMiss(f"offline mode and no recorded response for {url}")

        body = self._fetch_with_retries(url)
        self.cache.put(url, body)
        return body

    def _fetch_with_retries(self, url: str) -> Any:
        attempts = self.settings.ctgov_max_retries + 1
        last_error: str = ""
        for attempt in range(attempts):
            self._limiter.wait()
            try:
                response = self.http.get(url)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError:
                        raise CTGovError(f"non-JSON response from {url}") from None
                if response.status_code not in RETRYABLE_STATUS:
                    # 400s here mean we built a bad query (e.g. unknown field). Retrying won't help.
                    raise CTGovError(
                        f"ClinicalTrials.gov returned {response.status_code} for {url}: "
                        f"{response.text[:200]}"
                    )
                last_error = f"HTTP {response.status_code}"
                retry_after = _retry_after_seconds(response)
                if retry_after is not None and attempt < attempts - 1:
                    log.warning("rate limited by ClinicalTrials.gov; sleeping %.1fs", retry_after)
                    time.sleep(retry_after)
                    continue

            if attempt < attempts - 1:
                backoff = min(2 ** attempt, 20) + random.uniform(0, 0.5)
                log.warning("ClinicalTrials.gov request failed (%s); retrying in %.1fs", last_error, backoff)
                time.sleep(backoff)

        raise CTGovError(f"ClinicalTrials.gov unavailable after {attempts} attempts ({last_error})")


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return min(float(value), 60.0)
    except ValueError:
        return None
