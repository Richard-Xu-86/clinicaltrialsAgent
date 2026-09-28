"""The query plan: the only thing the language model is allowed to produce.

A plan is a small, closed description of *what to compute*. It contains no data
and no numbers other than filter values, so a bad plan can make us answer the
wrong question but can never make us report a count that isn't in the API data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import (
    AnalysisType,
    CompareField,
    Dimension,
    EntityType,
    NumericField,
    OverallStatus,
    Phase,
    StudyType,
)
from .request import MAX_YEAR, MIN_YEAR, parse_phase, parse_status


class PlanFilters(BaseModel):
    """Filters shared by every cohort in the plan."""

    model_config = ConfigDict(extra="forbid")

    drug: str | None = Field(None, description="Drug/intervention name, e.g. 'pembrolizumab'.")
    condition: str | None = Field(None, description="Condition/disease, e.g. 'breast cancer'.")
    sponsor: str | None = Field(None, description="Lead sponsor organisation, e.g. 'Pfizer'.")
    country: str | None = Field(None, description="Country name, e.g. 'Canada'.")
    phases: list[Phase] = Field(default_factory=list)
    statuses: list[OverallStatus] = Field(default_factory=list)
    study_type: StudyType | None = None
    start_year: int | None = Field(None, ge=MIN_YEAR, le=MAX_YEAR)
    end_year: int | None = Field(None, ge=MIN_YEAR, le=MAX_YEAR)

    # The model sometimes writes "Phase 3" instead of "PHASE3"; accept it rather than
    # burning a retry on a formatting nit.
    @field_validator("phases", mode="before")
    @classmethod
    def _phases(cls, value):
        return [parse_phase(v) if isinstance(v, str) else v for v in (value or [])]

    @field_validator("statuses", mode="before")
    @classmethod
    def _statuses(cls, value):
        return [parse_status(v) if isinstance(v, str) else v for v in (value or [])]


class CompareSpec(BaseModel):
    """Split the data into cohorts, one per value (e.g. drug A vs drug B)."""

    model_config = ConfigDict(extra="forbid")

    field: CompareField
    values: list[str] = Field(..., min_length=2, max_length=5)


class NetworkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: EntityType
    target: EntityType = Field(..., description="Same as source for co-occurrence (drug<->drug).")
    top_n_nodes: int = Field(25, ge=2, le=100)
    min_edge_weight: int | None = Field(
        None, ge=1,
        description="Drop edges supported by fewer trials. Default: 2 for co-occurrence "
                    "(same entity type), 1 for bipartite graphs.",
    )


class NumericSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: NumericField
    y: NumericField | None = Field(None, description="Required for scatter plots only.")


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: AnalysisType
    filters: PlanFilters = Field(default_factory=PlanFilters)
    group_by: Dimension | None = Field(
        None, description="Category axis for distribution/comparison; start_year for trend."
    )
    series_by: Dimension | None = Field(
        None, description="Optional second dimension, e.g. trend split by phase."
    )
    compare: CompareSpec | None = None
    network: NetworkSpec | None = None
    numeric: NumericSpec | None = None
    top_n: int | None = Field(None, ge=1, le=100)
    rationale: str | None = Field(
        None, max_length=400, description="One sentence on how the question was interpreted."
    )
