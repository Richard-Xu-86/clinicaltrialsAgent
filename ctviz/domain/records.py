"""Normalized trial records.

Every value we might count carries the JSON path it came from and the exact raw
text at that path. Citations are therefore a by-product of normalization rather
than something reconstructed afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..schemas import Evidence

T = TypeVar("T")


@dataclass(frozen=True)
class Sourced(Generic[T]):
    value: T      # normalized value used for grouping/arithmetic
    path: str     # JSON path in the API study record
    raw: str      # verbatim excerpt at that path

    def evidence(self) -> Evidence:
        return Evidence(field=self.path, value=self.raw)


@dataclass(frozen=True)
class PartialDate:
    year: int
    month: int | None
    day: int | None

    @property
    def month_index(self) -> int | None:
        return self.year * 12 + (self.month - 1) if self.month else None


@dataclass(frozen=True)
class Intervention:
    name: Sourced[str]
    type: str | None
    other_names: tuple[Sourced[str], ...]


@dataclass(frozen=True)
class Site:
    country: Sourced[str]
    status: str | None  # site-level recruitment status, when reported


@dataclass
class TrialRecord:
    nct_id: str
    title: str | None = None
    phase: Sourced[str] | None = None            # bucket key, e.g. "PHASE1/PHASE2"
    status: Sourced[str] | None = None
    study_type: Sourced[str] | None = None
    start: Sourced[PartialDate] | None = None
    primary_completion: Sourced[PartialDate] | None = None
    lead_sponsor: Sourced[str] | None = None
    sponsor_class: Sourced[str] | None = None
    enrollment: Sourced[int] | None = None
    interventions: list[Intervention] = field(default_factory=list)
    conditions: list[Sourced[str]] = field(default_factory=list)
    sites: list[Site] = field(default_factory=list)
    # Evidence that the trial matches the request's scope (e.g. the intervention entry
    # naming the requested drug). Filled in by verification; appended to citations.
    match_evidence: list[Evidence] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"
