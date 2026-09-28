"""Request schema for POST /v1/visualize."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import OverallStatus, Phase, StudyType

MIN_YEAR = 1990
MAX_YEAR = 2040

# Loose spellings people actually type -> CT.gov phase codes.
_PHASE_ALIASES = {
    "0": Phase.EARLY_PHASE1,
    "early1": Phase.EARLY_PHASE1,
    "earlyphase1": Phase.EARLY_PHASE1,
    "1": Phase.PHASE1,
    "i": Phase.PHASE1,
    "2": Phase.PHASE2,
    "ii": Phase.PHASE2,
    "3": Phase.PHASE3,
    "iii": Phase.PHASE3,
    "4": Phase.PHASE4,
    "iv": Phase.PHASE4,
    "na": Phase.NA,
    "notapplicable": Phase.NA,
}


def parse_phase(value: str) -> Phase:
    """Accept 'PHASE3', 'Phase 3', 'phase iii', '3', ... and return the CT.gov code."""
    raw = value.strip()
    try:
        return Phase(raw.upper())
    except ValueError:
        pass
    key = re.sub(r"[\s_\-]", "", raw.lower()).removeprefix("phase")
    if key in _PHASE_ALIASES:
        return _PHASE_ALIASES[key]
    raise ValueError(f"unrecognised trial phase {value!r}; expected e.g. 'PHASE2' or 'Phase 2'")


def parse_status(value: str) -> OverallStatus:
    key = re.sub(r"[\s\-,]+", "_", value.strip().upper())
    try:
        return OverallStatus(key)
    except ValueError:
        allowed = ", ".join(s.value for s in OverallStatus)
        raise ValueError(f"unrecognised status {value!r}; allowed: {allowed}") from None


class VisualizeRequest(BaseModel):
    """A natural-language question plus optional structured filters.

    Structured fields are authoritative: when both the question and a field
    mention a drug, the field wins. That gives API callers a deterministic way to
    pin down what the language model might otherwise misread.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., min_length=3, max_length=500, description="Natural-language question.")

    drug_name: str | None = Field(None, max_length=100, description="Intervention/drug to scope to.")
    condition: str | None = Field(None, max_length=100, description="Condition or disease.")
    sponsor: str | None = Field(None, max_length=150, description="Lead sponsor name.")
    country: str | None = Field(None, max_length=60, description="Country with at least one site.")
    trial_phase: list[Phase] | None = Field(
        None, description="One or more phases. Accepts 'PHASE3', 'Phase 3', '3', ..."
    )
    status: list[OverallStatus] | None = Field(
        None, description="Overall recruitment status(es), e.g. RECRUITING."
    )
    study_type: StudyType | None = None
    start_year: int | None = Field(None, ge=MIN_YEAR, le=MAX_YEAR, description="Inclusive.")
    end_year: int | None = Field(None, ge=MIN_YEAR, le=MAX_YEAR, description="Inclusive.")

    max_records: int = Field(
        3000, ge=50, le=10000,
        description="Upper bound on trial records fetched per cohort. Larger = slower but less truncation.",
    )
    max_citations_per_datum: int = Field(
        10, ge=0, le=1000,
        description="How many citations to attach to each bar/bucket/node/edge. The full count is always reported.",
    )
    top_n: int | None = Field(
        None, ge=1, le=100, description="Keep only the N largest categories (or nodes, for networks)."
    )

    @field_validator("trial_phase", mode="before")
    @classmethod
    def _coerce_phases(cls, value):
        if value is None:
            return None
        items = value if isinstance(value, list) else [value]
        return [parse_phase(v) if isinstance(v, str) else v for v in items]

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_statuses(cls, value):
        if value is None:
            return None
        items = value if isinstance(value, list) else [value]
        return [parse_status(v) if isinstance(v, str) else v for v in items]

    @field_validator("study_type", mode="before")
    @classmethod
    def _coerce_study_type(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("drug_name", "condition", "sponsor", "country")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        return value or None

    @model_validator(mode="after")
    def _check_year_range(self) -> VisualizeRequest:
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        return self
