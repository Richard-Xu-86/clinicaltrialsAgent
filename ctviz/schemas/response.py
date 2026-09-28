"""Response schema: a renderer-agnostic visualization spec plus metadata.

Contract for frontend engineers:

* ``visualization.type`` selects the renderer and the shape of ``data``.
* For every type except ``network_graph``, ``data`` is a list of flat rows. Each
  channel in ``encoding`` names the row key it reads (``encoding.x.field`` etc.).
  Rows are already sorted in display order; no client-side sorting is needed.
* For ``network_graph``, ``data`` is ``{"nodes": [...], "edges": [...]}`` and
  ``encoding`` says which node/edge keys carry id, label, size, colour and weight.
* Every row, node and edge carries ``citations`` (possibly truncated) and
  ``citation_count`` (never truncated). Renderers can ignore both.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .enums import VisualizationType
from .plan import QueryPlan

# --------------------------------------------------------------------------- citations

class Evidence(BaseModel):
    field: str = Field(..., description="JSON path inside the study record, e.g. protocolSection.designModule.phases")
    value: str | None = Field(
        ..., description="Exact value at that path, verbatim from the API response. null means the field "
                         "is absent from the record (used for 'not reported' buckets).",
    )


class Citation(BaseModel):
    nct_id: str
    title: str | None = None
    url: str
    evidence: list[Evidence] = Field(..., description="Field/value pairs that justify counting this trial here.")


# --------------------------------------------------------------------------- encoding

ChannelType = Literal["nominal", "ordinal", "quantitative", "temporal"]


class Channel(BaseModel):
    field: str
    type: ChannelType
    title: str
    unit: str | None = None
    sort: list[str] | Literal["ascending", "descending"] | None = Field(
        None, description="Explicit category order, or a direction. Data already follows it."
    )


class CartesianEncoding(BaseModel):
    x: Channel
    y: Channel
    series: Channel | None = Field(None, description="Colour/grouping channel (grouped bars, multi-line).")
    tooltip: list[str] = Field(default_factory=list, description="Row keys worth showing on hover.")


class NodeEncoding(BaseModel):
    id: str = "id"
    label: str = "label"
    size: str = "trial_count"
    color: str = "entity_type"


class EdgeEncoding(BaseModel):
    source: str = "source"
    target: str = "target"
    weight: str = "trial_count"


class NetworkEncoding(BaseModel):
    nodes: NodeEncoding = Field(default_factory=NodeEncoding)
    edges: EdgeEncoding = Field(default_factory=EdgeEncoding)
    directed: bool = False


# --------------------------------------------------------------------------- data

class NetworkNode(BaseModel):
    id: str
    label: str
    entity_type: str
    trial_count: int
    citation_count: int
    citations: list[Citation]


class NetworkEdge(BaseModel):
    source: str
    target: str
    trial_count: int
    citation_count: int
    citations: list[Citation]


class NetworkData(BaseModel):
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]


Row = dict[str, Any]


class CartesianVisualization(BaseModel):
    type: Literal[
        VisualizationType.BAR_CHART,
        VisualizationType.GROUPED_BAR_CHART,
        VisualizationType.TIME_SERIES,
        VisualizationType.HISTOGRAM,
        VisualizationType.SCATTER_PLOT,
    ]
    title: str
    subtitle: str | None = None
    encoding: CartesianEncoding
    data: list[Row] = Field(
        ..., description="Flat rows keyed by the encoding fields, plus citations and citation_count."
    )


class NetworkVisualization(BaseModel):
    type: Literal[VisualizationType.NETWORK_GRAPH]
    title: str
    subtitle: str | None = None
    encoding: NetworkEncoding = Field(default_factory=NetworkEncoding)
    data: NetworkData


Visualization = Annotated[
    CartesianVisualization | NetworkVisualization, Field(discriminator="type")
]


# --------------------------------------------------------------------------- meta

class CohortStats(BaseModel):
    label: str
    api_total: int = Field(..., description="totalCount reported by ClinicalTrials.gov for this cohort's query.")
    fetched: int = Field(..., description="Records actually downloaded (capped by max_records).")
    analyzed: int = Field(..., description="Records used after verification.")
    excluded: int = Field(0, description="Fetched records dropped by verification (see counting_rules).")
    excluded_nct_ids: list[str] = Field(default_factory=list, description="Up to 25 excluded trials, for auditing.")
    truncated: bool


class PlannerInfo(BaseModel):
    method: Literal["llm", "rule_based"]
    model: str | None = None
    fallback_reason: str | None = None
    plan: QueryPlan
    overrides: list[str] = Field(default_factory=list, description="Plan fields replaced by explicit request fields.")
    adjustments: list[str] = Field(default_factory=list, description="Fixes applied to make the plan executable.")


class ResponseMeta(BaseModel):
    source: str = "ClinicalTrials.gov API v2"
    api_version: str | None = None
    data_timestamp: str | None = None
    query: str
    planner: PlannerInfo
    filters_applied: dict[str, Any]
    cohorts: list[CohortStats]
    units: str = "trials"
    time_granularity: Literal["year"] | None = None
    counting_rules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    api_requests: list[str] = Field(default_factory=list, description="API URLs queried (first page of each).")
    generated_at: datetime


class VisualizeResponse(BaseModel):
    visualization: Visualization
    meta: ResponseMeta
