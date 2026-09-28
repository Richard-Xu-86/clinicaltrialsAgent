"""The request pipeline: plan -> fetch -> normalize -> verify -> aggregate -> spec.

Only the first step involves a language model. Everything after it is ordinary,
deterministic code working on API records, which is where every number and
every citation in the response comes from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .analysis.charts import (
    ENTITY_DIMENSION,
    BuildOptions,
    BuildResult,
    CohortData,
    build_comparison,
    build_distribution,
    build_histogram,
    build_network,
    build_scatter,
    build_trend,
)
from .analysis.dimensions import DIMENSIONS, INTERVENTION_FIELDS, NUMERIC_FIELDS, ExtractContext
from .config import Settings, get_settings
from .ctgov.client import CTGovClient, CTGovError, OfflineCacheMiss
from .ctgov.query import build_search_params
from .domain.drugs import display_name, find_drug_evidence
from .domain.normalize import normalize_studies
from .domain.records import TrialRecord
from .planner import OpenAIPlanner, plan_request
from .schemas import (
    AnalysisType,
    CohortStats,
    CompareField,
    OverallStatus,
    PlanFilters,
    QueryPlan,
    ResponseMeta,
    VisualizeRequest,
    VisualizeResponse,
)

log = logging.getLogger(__name__)

BASE_FIELDS = ("NCTId", "BriefTitle")


class UpstreamError(RuntimeError):
    """ClinicalTrials.gov failed; mapped to HTTP 502 by the API layer."""


@dataclass
class CohortPlan:
    label: str
    filters: PlanFilters


def expand_cohorts(plan: QueryPlan) -> list[CohortPlan]:
    """One cohort per compared value, or a single cohort for the shared filters."""
    if plan.compare is None:
        return [CohortPlan(label=_scope_label(plan.filters), filters=plan.filters)]
    cohorts = []
    for value in plan.compare.values:
        filters = plan.filters.model_copy(update={plan.compare.field.value: value})
        label = display_name(value.lower()) if plan.compare.field == CompareField.DRUG else value
        cohorts.append(CohortPlan(label=label, filters=filters))
    return cohorts


def _scope_label(filters: PlanFilters) -> str:
    if filters.drug:
        return display_name(filters.drug.lower())
    return filters.condition or filters.sponsor or filters.country or "All trials"


def required_fields(plan: QueryPlan, cohorts: list[CohortPlan]) -> set[str]:
    """Request only the API fields the analysis needs; keeps pages small and fast."""
    fields = set(BASE_FIELDS)
    for dim_key in (plan.group_by, plan.series_by):
        if dim_key is not None:
            fields |= set(DIMENSIONS[dim_key].api_fields)
    if plan.analysis == AnalysisType.TREND:
        fields |= {"StartDate"}
    if plan.network:
        for entity in (plan.network.source, plan.network.target):
            fields |= set(DIMENSIONS[ENTITY_DIMENSION[entity]].api_fields)
    if plan.numeric:
        for numeric in (plan.numeric.x, plan.numeric.y):
            if numeric is not None:
                fields |= set(NUMERIC_FIELDS[numeric].api_fields)
    if any(c.filters.drug for c in cohorts):
        fields |= set(INTERVENTION_FIELDS)  # needed to verify drug matches
    return fields


def verify_cohort(records: list[TrialRecord], filters: PlanFilters) -> tuple[list[TrialRecord], list[str]]:
    """Keep only trials whose own record proves they match the drug filter.

    ``query.intr`` also matches trials that mention a drug only in the
    description or keywords (or as "placebo for X"). Those are excluded, so every
    counted trial can be cited with the intervention entry that names the drug.
    """
    if not filters.drug:
        return records, []
    kept, excluded = [], []
    for rec in records:
        evidence = find_drug_evidence(rec, filters.drug)
        if evidence is None:
            excluded.append(rec.nct_id)
        else:
            rec.match_evidence.append(evidence.evidence())
            kept.append(rec)
    return kept, excluded


class VisualizationService:
    def __init__(self, settings: Settings | None = None, client: CTGovClient | None = None,
                 llm: OpenAIPlanner | None = None):
        self.settings = settings or get_settings()
        self.client = client or CTGovClient(self.settings)
        self.llm = llm

    def visualize(self, request: VisualizeRequest) -> VisualizeResponse:
        planner = plan_request(request, self.settings, self.llm)
        plan = planner.plan
        cohort_plans = expand_cohorts(plan)
        fields = required_fields(plan, cohort_plans)

        cohorts: list[CohortData] = []
        stats: list[CohortStats] = []
        api_requests: list[str] = []
        warnings: list[str] = []
        for cp in cohort_plans:
            params = build_search_params(cp.filters, fields)
            try:
                result = self.client.search(params, request.max_records)
            except OfflineCacheMiss:
                raise  # a configuration problem, not an upstream failure; the API maps it to 503
            except CTGovError as exc:
                raise UpstreamError(str(exc)) from exc
            api_requests.append(result.first_page_url)

            records = normalize_studies(result.studies)
            kept, excluded = verify_cohort(records, cp.filters)
            cohorts.append(CohortData(cp.label, cp.filters, kept))
            stats.append(CohortStats(
                label=cp.label, api_total=result.total_count, fetched=len(records), analyzed=len(kept),
                excluded=len(excluded), excluded_nct_ids=sorted(excluded)[:25], truncated=result.truncated,
            ))
            if result.truncated:
                warnings.append(
                    f"{cp.label}: ClinicalTrials.gov matched {result.total_count} trials; only the first "
                    f"{len(records)} were analyzed (raise max_records to include more)."
                )

        if not any(c.records for c in cohorts):
            warnings.append("No trials matched the query; the visualization is empty. "
                            "Check spelling or loosen the filters.")

        only_recruiting = plan.filters.statuses == [OverallStatus.RECRUITING]
        opts = BuildOptions(
            citation_limit=request.max_citations_per_datum,
            top_n=plan.top_n,
            ctx=ExtractContext(recruiting_sites_only=only_recruiting),
        )
        built = self._build(plan, cohorts, opts)

        rules = list(built.counting_rules)
        if any(c.filters.drug for c in cohorts):
            rules.insert(0, "Drug filter: ClinicalTrials.gov search results were kept only if an "
                            "intervention name, a registered synonym of it, or the brief title names the drug "
                            "(placebo arms don't count). The matching field is included in each citation; "
                            "dropped trials are listed in meta.cohorts[].excluded_nct_ids.")
        if only_recruiting and plan.group_by and plan.group_by.value == "country":
            rules.append("Only sites whose own status is RECRUITING count toward a country.")

        meta = ResponseMeta(
            query=request.query,
            planner=planner,
            filters_applied=_describe_filters(plan),
            cohorts=stats,
            time_granularity=built.time_granularity,
            counting_rules=rules,
            warnings=warnings + built.warnings,
            api_requests=api_requests,
            generated_at=datetime.now(timezone.utc),
            **self._provenance(),
        )
        return VisualizeResponse(visualization=built.visualization, meta=meta)

    # ------------------------------------------------------------------ helpers

    def _build(self, plan: QueryPlan, cohorts: list[CohortData], opts: BuildOptions) -> BuildResult:
        if plan.analysis == AnalysisType.TREND:
            series_dim = DIMENSIONS[plan.series_by] if plan.series_by else None
            series_title = (series_dim.title if series_dim
                            else plan.compare.field.value.capitalize() if plan.compare else None)
            return build_trend(cohorts, series_dim, series_title, opts)
        if plan.analysis == AnalysisType.DISTRIBUTION:
            return build_distribution(cohorts[0], DIMENSIONS[plan.group_by], opts)
        if plan.analysis == AnalysisType.COMPARISON:
            return build_comparison(cohorts, plan.filters, DIMENSIONS[plan.group_by],
                                    plan.compare.field.value.capitalize(), opts)
        if plan.analysis == AnalysisType.NETWORK:
            return build_network(cohorts, plan.network, opts)
        if plan.analysis == AnalysisType.HISTOGRAM:
            return build_histogram(cohorts, plan.numeric.x, opts)
        if plan.analysis == AnalysisType.SCATTER:
            return build_scatter(cohorts, plan.numeric.x, plan.numeric.y, opts)
        raise ValueError(f"unsupported analysis {plan.analysis}")  # unreachable: enum is closed

    def _provenance(self) -> dict:
        try:
            version = self.client.version()
            return {"api_version": version.get("apiVersion"), "data_timestamp": version.get("dataTimestamp")}
        except CTGovError as exc:
            log.info("could not read API version: %s", exc)
            return {}


def _describe_filters(plan: QueryPlan) -> dict:
    filters = plan.filters.model_dump(mode="json", exclude_none=True)
    filters = {k: v for k, v in filters.items() if v != []}
    if plan.compare:
        filters["compare"] = plan.compare.model_dump(mode="json")
    return filters
