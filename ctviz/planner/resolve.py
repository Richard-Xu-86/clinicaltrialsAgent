"""Make a plan executable: apply explicit request fields and fix inconsistencies.

Schema validation (pydantic) already guarantees the plan only uses known enums.
This step checks the *combination* of choices, e.g. a comparison with a single
cohort, or a trend grouped by something other than time. Each fix is recorded so
the response can show exactly how the question was interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import (
    AnalysisType,
    CompareField,
    Dimension,
    EntityType,
    NetworkSpec,
    NumericField,
    NumericSpec,
    QueryPlan,
    VisualizeRequest,
)

# Explicit request field -> plan filter field
_FIELD_MAP = {
    "drug_name": "drug",
    "condition": "condition",
    "sponsor": "sponsor",
    "country": "country",
    "trial_phase": "phases",
    "status": "statuses",
    "study_type": "study_type",
    "start_year": "start_year",
    "end_year": "end_year",
}

# Grouping a comparison of drugs by drug (etc.) would be meaningless.
_COMPARE_DIMENSION = {
    CompareField.DRUG: Dimension.DRUG,
    CompareField.CONDITION: Dimension.CONDITION,
    CompareField.SPONSOR: Dimension.LEAD_SPONSOR,
    CompareField.COUNTRY: Dimension.COUNTRY,
}


@dataclass
class Resolution:
    plan: QueryPlan
    overrides: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)
    explicit: set[str] = field(default_factory=set)  # plan filter fields set from the request


def _same(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.casefold().strip() == b.casefold().strip()
    if isinstance(a, list) and isinstance(b, list):
        return sorted(map(str, a)) == sorted(map(str, b))
    return a == b


def resolve_plan(plan: QueryPlan, request: VisualizeRequest) -> Resolution:
    plan = plan.model_copy(deep=True)
    res = Resolution(plan)
    _apply_explicit_fields(res, request)
    _fix_years(res)
    _fix_analysis(res)
    return res


def _apply_explicit_fields(res: Resolution, request: VisualizeRequest) -> None:
    plan = res.plan
    for req_field, plan_field in _FIELD_MAP.items():
        explicit = getattr(request, req_field)
        if explicit in (None, []):
            continue
        current = getattr(plan.filters, plan_field)

        # "Compare this drug with nivolumab" + drug_name=pembrolizumab: the explicit
        # value belongs in the comparison, not in a filter shared by both cohorts.
        if plan.compare and plan.compare.field.value == plan_field and isinstance(explicit, str):
            if not any(_same(explicit, v) for v in plan.compare.values):
                if len(plan.compare.values) >= 5:
                    dropped = plan.compare.values.pop()
                    res.overrides.append(f"compare.values: dropped {dropped!r} (max 5 cohorts) to make room")
                plan.compare.values.insert(0, explicit)
                res.overrides.append(f"compare.values: added {req_field}={explicit!r}")
            continue

        if current not in (None, []) and not _same(current, explicit):
            res.overrides.append(f"filters.{plan_field}: {current!r} -> {explicit!r} (from {req_field})")
        setattr(plan.filters, plan_field, explicit)
        res.explicit.add(plan_field)

    if request.top_n is not None:
        plan.top_n = request.top_n
        if plan.network:
            plan.network.top_n_nodes = max(2, request.top_n)


def _fix_years(res: Resolution) -> None:
    f = res.plan.filters
    if not (f.start_year and f.end_year and f.start_year > f.end_year):
        return
    # Request validation guarantees explicit years are ordered, so a conflict involves at
    # least one planner-supplied year. Never move an explicit value; drop the planner's.
    if "start_year" in res.explicit:
        res.adjustments.append(f"Dropped planner end_year {f.end_year}: it precedes the requested start_year.")
        f.end_year = None
    elif "end_year" in res.explicit:
        res.adjustments.append(f"Dropped planner start_year {f.start_year}: it follows the requested end_year.")
        f.start_year = None
    else:
        f.start_year, f.end_year = f.end_year, f.start_year
        res.adjustments.append("start_year and end_year were reversed; swapped them.")


def _fix_analysis(res: Resolution) -> None:
    plan = res.plan
    adjust = res.adjustments.append

    # A compare spec whose field is also a shared filter would make every cohort identical.
    if plan.compare:
        shared = getattr(plan.filters, plan.compare.field.value)
        if shared:
            adjust(f"Removed shared {plan.compare.field.value} filter {shared!r}; the comparison "
                   "defines it per cohort.")
            setattr(plan.filters, plan.compare.field.value, None)
        # de-duplicate compare values, case-insensitively
        seen, values = set(), []
        for v in plan.compare.values:
            if v.casefold() not in seen:
                seen.add(v.casefold())
                values.append(v)
        plan.compare.values = values
        if len(values) < 2:
            adjust("Comparison needs at least two distinct values; showing a single distribution instead.")
            setattr(plan.filters, plan.compare.field.value, values[0] if values else None)
            plan.compare = None

    if plan.analysis == AnalysisType.DISTRIBUTION and plan.group_by == Dimension.START_YEAR:
        adjust("Grouping by start year is a time trend; switched analysis to 'trend'.")
        plan.analysis = AnalysisType.TREND

    if plan.analysis == AnalysisType.DISTRIBUTION and plan.compare:
        adjust("A distribution with cohorts is a comparison; switched analysis to 'comparison'.")
        plan.analysis = AnalysisType.COMPARISON

    if plan.analysis == AnalysisType.COMPARISON and not plan.compare:
        adjust("No cohorts to compare were identified; showing a single distribution.")
        plan.analysis = AnalysisType.DISTRIBUTION

    if plan.analysis in (AnalysisType.DISTRIBUTION, AnalysisType.COMPARISON):
        if plan.group_by is None:
            plan.group_by = Dimension.PHASE
            adjust("No grouping dimension given; defaulted to phase.")
        if plan.compare and plan.group_by == _COMPARE_DIMENSION[plan.compare.field]:
            adjust(f"Cannot group a {plan.compare.field.value} comparison by {plan.group_by.value}; "
                   "grouped by phase instead.")
            plan.group_by = Dimension.PHASE
        plan.series_by = None

    elif plan.analysis == AnalysisType.TREND:
        # "Trend of phases over time" -> years on x, phase as the series.
        if plan.group_by not in (None, Dimension.START_YEAR) and plan.series_by is None and not plan.compare:
            plan.series_by = plan.group_by
            adjust(f"Trend uses start year on the x-axis; {plan.group_by.value} became the series.")
        plan.group_by = Dimension.START_YEAR
        if plan.series_by == Dimension.START_YEAR:
            plan.series_by = None
        if plan.compare and plan.series_by:
            adjust("Trend with cohorts already uses cohorts as series; dropped series_by.")
            plan.series_by = None

    elif plan.analysis == AnalysisType.NETWORK:
        if plan.network is None:
            if plan.filters.condition or plan.filters.country:
                plan.network = NetworkSpec(source=EntityType.SPONSOR, target=EntityType.DRUG)
            else:
                plan.network = NetworkSpec(source=EntityType.DRUG, target=EntityType.DRUG)
            adjust(f"No network entities given; defaulted to {plan.network.source.value}–"
                   f"{plan.network.target.value}.")
        if plan.top_n is not None:
            plan.network.top_n_nodes = max(2, plan.top_n)
        if plan.network.min_edge_weight is None:
            # A pair of drugs used together once is noise; a sponsor running one trial of a
            # drug is a real (and typical) relationship.
            same = plan.network.source == plan.network.target
            plan.network.min_edge_weight = 2 if same else 1

    elif plan.analysis == AnalysisType.HISTOGRAM:
        if plan.numeric is None:
            plan.numeric = NumericSpec(x=NumericField.ENROLLMENT)
            adjust("No numeric field given for the histogram; defaulted to enrollment.")
        plan.numeric.y = None

    elif plan.analysis == AnalysisType.SCATTER:
        if plan.numeric is None:
            plan.numeric = NumericSpec(x=NumericField.START_YEAR, y=NumericField.ENROLLMENT)
            adjust("No axes given for the scatter plot; defaulted to start year vs enrollment.")
        if plan.numeric.y is None or plan.numeric.y == plan.numeric.x:
            plan.numeric.y = (NumericField.ENROLLMENT if plan.numeric.x != NumericField.ENROLLMENT
                              else NumericField.DURATION_MONTHS)
            adjust(f"Scatter plot needs two different numeric fields; using y={plan.numeric.y.value}.")

    # Fields that only make sense for other analysis types are cleared, so the plan
    # echoed in the response describes what was actually computed.
    if plan.analysis != AnalysisType.NETWORK:
        plan.network = None
    if plan.analysis not in (AnalysisType.HISTOGRAM, AnalysisType.SCATTER):
        plan.numeric = None
    if plan.analysis in (AnalysisType.NETWORK, AnalysisType.HISTOGRAM, AnalysisType.SCATTER):
        plan.group_by = None
        plan.series_by = None
