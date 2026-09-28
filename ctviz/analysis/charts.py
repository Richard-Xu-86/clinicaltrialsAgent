"""Turn aggregated buckets into visualization specs.

One builder per analysis type. Builders are deterministic: the same records and
plan always produce the same JSON, which is what makes the example outputs and
the tests meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from ..domain.drugs import display_name
from ..domain.records import TrialRecord
from ..schemas import (
    CartesianEncoding,
    CartesianVisualization,
    Channel,
    Dimension,
    EntityType,
    NetworkData,
    NetworkEdge,
    NetworkNode,
    NetworkSpec,
    NetworkVisualization,
    NumericField,
    PlanFilters,
    VisualizationType,
)
from .aggregate import (
    Bucket,
    Grouping,
    co_occurrence,
    equal_width_edges,
    group_by,
    make_citation,
    numeric_points,
    ordered_buckets,
)
from .dimensions import DIMENSIONS, NUMERIC_FIELDS, DimensionDef, ExtractContext, missing_description

COUNT_FIELD = "trial_count"
COUNT_CHANNEL = Channel(field=COUNT_FIELD, type="quantitative", title="Number of trials", unit="trials")

# Dimensions with open-ended cardinality get a default top-N so charts stay readable.
DEFAULT_TOP_N = {Dimension.DRUG: 15, Dimension.CONDITION: 15, Dimension.LEAD_SPONSOR: 15, Dimension.COUNTRY: 15}
DEFAULT_TOP_N_SERIES = 8  # more lines than this on one time series is unreadable

ENTITY_DIMENSION = {
    EntityType.DRUG: Dimension.DRUG,
    EntityType.SPONSOR: Dimension.LEAD_SPONSOR,
    EntityType.CONDITION: Dimension.CONDITION,
    EntityType.COUNTRY: Dimension.COUNTRY,
}


@dataclass
class CohortData:
    label: str
    filters: PlanFilters
    records: list[TrialRecord]


@dataclass
class BuildOptions:
    citation_limit: int
    top_n: int | None
    ctx: ExtractContext


@dataclass
class BuildResult:
    visualization: CartesianVisualization | NetworkVisualization
    counting_rules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    time_granularity: str | None = None


# ----------------------------------------------------------------------------- helpers

def describe_scope(filters: PlanFilters) -> str:
    parts = []
    if filters.drug:
        parts.append(display_name(filters.drug.lower()))
    if filters.condition:
        parts.append(filters.condition)
    if filters.sponsor:
        parts.append(f"sponsored by {filters.sponsor}")
    if filters.country:
        parts.append(f"in {filters.country}")
    if filters.phases:
        parts.append("/".join(p.value.replace("PHASE", "Phase ").replace("EARLY_", "Early ") for p in filters.phases))
    if filters.statuses:
        parts.append(", ".join(s.value.replace("_", " ").lower() for s in filters.statuses))
    if filters.start_year or filters.end_year:
        parts.append(f"{filters.start_year or '…'}–{filters.end_year or 'present'}")
    return "; ".join(parts) if parts else "all registered trials"


def _cartesian(vtype, title, subtitle, encoding, rows) -> CartesianVisualization:
    return CartesianVisualization(type=vtype, title=title, subtitle=subtitle, encoding=encoding, data=rows)


def _row(bucket: Bucket, limit: int, **fields) -> dict:
    return {**fields, COUNT_FIELD: bucket.count, "citation_count": bucket.count, "citations": bucket.citations(limit)}


def _empty_row(**fields) -> dict:
    return {**fields, COUNT_FIELD: 0, "citation_count": 0, "citations": []}


def _apply_top_n(buckets: list[Bucket], dim: DimensionDef, top_n: int | None, warnings: list[str],
                 default: int | None = None) -> list[Bucket]:
    if dim.order:
        if top_n:
            warnings.append(f"top_n ignored: {dim.title.lower()} has a fixed set of categories.")
        return buckets
    limit = top_n or (default if default is not None else DEFAULT_TOP_N.get(dim.key))
    if limit is None or len(buckets) <= limit:
        return buckets
    warnings.append(f"Showing the top {limit} of {len(buckets)} {dim.title.lower()} categories.")
    by_size = sorted(buckets, key=lambda b: -b.count)[:limit]
    keep = {b.key for b in by_size}
    return [b for b in buckets if b.key in keep]


def _rules_for(dim: DimensionDef) -> list[str]:
    return [dim.counting_rule] if dim.counting_rule else []


def _missing_warning(missing: int, total: int, what: str) -> list[str]:
    if not missing:
        return []
    return [f"{missing} of {total} trials have no {what} and are not shown."]


# ----------------------------------------------------------------------------- distribution

def build_distribution(cohort: CohortData, dim: DimensionDef, opts: BuildOptions) -> BuildResult:
    grouping = group_by(cohort.records, dim, opts.ctx)
    warnings = _missing_warning(grouping.missing, len(cohort.records), missing_description(dim, opts.ctx))
    buckets = _apply_top_n(ordered_buckets(grouping, dim), dim, opts.top_n, warnings)

    rows = [_row(b, opts.citation_limit, **{dim.row_field: b.label}) for b in buckets]
    encoding = CartesianEncoding(
        x=Channel(field=dim.row_field, type="ordinal" if dim.order else "nominal", title=dim.title,
                  sort=[b.label for b in buckets]),
        y=COUNT_CHANNEL,
        tooltip=[dim.row_field, COUNT_FIELD],
    )
    title = f"Trials by {dim.title.lower()}: {describe_scope(cohort.filters)}"
    subtitle = f"{len(cohort.records)} trials analyzed"
    rules = _rules_for(dim)
    if dim.multi_valued:
        rules.append("Because a trial can fall into several categories, bar heights can sum to more "
                     "than the number of trials.")
    return BuildResult(_cartesian(VisualizationType.BAR_CHART, title, subtitle, encoding, rows), rules, warnings)


# ----------------------------------------------------------------------------- comparison

def build_comparison(cohorts: list[CohortData], shared_filters: PlanFilters, dim: DimensionDef,
                     compare_title: str, opts: BuildOptions) -> BuildResult:
    warnings: list[str] = []
    per_cohort: list[tuple[CohortData, dict[str, Bucket]]] = []
    totals: dict[str, Bucket] = {}  # merged, only used for category ordering / top-N

    for cohort in cohorts:
        grouping = group_by(cohort.records, dim, opts.ctx)
        warnings += [f"{cohort.label}: {w}" for w in
                     _missing_warning(grouping.missing, len(cohort.records), missing_description(dim, opts.ctx))]
        per_cohort.append((cohort, grouping.buckets))
        for key, bucket in grouping.buckets.items():
            merged = totals.setdefault(key, Bucket(key, bucket.label))
            for rec, ev in bucket.trials.values():
                merged.add(rec, ev)

    categories = _apply_top_n(ordered_buckets(Grouping(totals, 0), dim), dim, opts.top_n, warnings)

    rows = []
    for bucket in categories:
        for cohort, buckets in per_cohort:
            n = len(cohort.records)
            b = buckets.get(bucket.key)
            fields = {dim.row_field: bucket.label, "cohort": cohort.label}
            row = _empty_row(**fields) if b is None else _row(b, opts.citation_limit, **fields)
            # Cohorts can differ a lot in size (668 semaglutide vs 250 tirzepatide trials in the
            # examples), so the share is often the fairer comparison. Renderers can switch y to it.
            row["share_of_cohort"] = round(row[COUNT_FIELD] / n, 4) if n else 0.0
            rows.append(row)

    encoding = CartesianEncoding(
        x=Channel(field=dim.row_field, type="ordinal" if dim.order else "nominal", title=dim.title,
                  sort=[b.label for b in categories]),
        y=COUNT_CHANNEL,
        series=Channel(field="cohort", type="nominal", title=compare_title, sort=[c.label for c in cohorts]),
        tooltip=[dim.row_field, "cohort", COUNT_FIELD, "share_of_cohort"],
    )
    labels = " vs ".join(c.label for c in cohorts)
    shared = describe_scope(shared_filters)
    title = f"Trials by {dim.title.lower()}: {labels}"
    subtitle = ", ".join(f"{c.label}: {len(c.records)} trials" for c in cohorts)
    if shared != "all registered trials":
        subtitle += f" ({shared})"
    rules = _rules_for(dim) + ["share_of_cohort = trial_count / trials analyzed in that cohort."]
    return BuildResult(_cartesian(VisualizationType.GROUPED_BAR_CHART, title, subtitle, encoding, rows),
                       rules, warnings)


# ----------------------------------------------------------------------------- trend

def build_trend(cohorts: list[CohortData], series_dim: DimensionDef | None, series_title: str | None,
                opts: BuildOptions) -> BuildResult:
    year_dim = DIMENSIONS[Dimension.START_YEAR]
    warnings: list[str] = []
    series: list[tuple[str | None, dict[str, Bucket]]] = []

    for cohort in cohorts:
        if series_dim is not None:
            # Trend split by a second dimension (e.g. per-year counts for each phase).
            by_series = group_by(cohort.records, series_dim, opts.ctx)
            warnings += _missing_warning(by_series.missing, len(cohort.records),
                                         missing_description(series_dim, opts.ctx))
            subs = _apply_top_n(ordered_buckets(by_series, series_dim), series_dim, opts.top_n, warnings,
                                default=DEFAULT_TOP_N_SERIES)
            undated: set[str] = set()
            for sub in subs:
                recs = [r for r, _ in sub.trials.values()]
                undated |= {r.nct_id for r in recs if r.start is None}
                series.append((sub.label, group_by(recs, year_dim, opts.ctx).buckets))
            warnings += _missing_warning(len(undated), len(cohort.records), "start date")
        else:
            grouping = group_by(cohort.records, year_dim, opts.ctx)
            prefix = f"{cohort.label}: " if len(cohorts) > 1 else ""
            warnings += [prefix + w for w in _missing_warning(grouping.missing, len(cohort.records), "start date")]
            series.append((cohort.label if len(cohorts) > 1 else None, grouping.buckets))

    years = sorted({int(y) for _, buckets in series for y in buckets})
    filters = cohorts[0].filters
    lo = filters.start_year or (years[0] if years else None)
    hi = filters.end_year or (years[-1] if years else None)
    rows = []
    if lo is not None and hi is not None:
        for label, buckets in series:
            for year in range(lo, hi + 1):  # zero-fill so gaps render as 0, not as a missing point
                fields = {"start_year": year}
                if label is not None:
                    fields["series"] = label
                b = buckets.get(str(year))
                rows.append(_row(b, opts.citation_limit, **fields) if b else _empty_row(**fields))

    this_year = date.today().year
    if hi is not None and hi >= this_year:
        warnings.append(f"{this_year} is incomplete, and later years only contain trials with an "
                        "estimated (planned) start date.")

    has_series = any(label is not None for label, _ in series)
    encoding = CartesianEncoding(
        # Years are plain integers; "ordinal" renders them as-is in any charting library,
        # whereas "temporal" invites parsing 2019 as a timestamp.
        x=Channel(field="start_year", type="ordinal", title="Start year", unit="year", sort="ascending"),
        y=COUNT_CHANNEL,
        series=Channel(field="series", type="nominal", title=series_title or "Series",
                       sort=list(dict.fromkeys(label for label, _ in series))) if has_series else None,
        tooltip=["start_year", COUNT_FIELD] + (["series"] if has_series else []),
    )
    scope = " vs ".join(c.label for c in cohorts) if len(cohorts) > 1 else describe_scope(filters)
    title = f"Trials started per year: {scope}"
    if series_dim is not None:
        title += f", by {series_dim.title.lower()}"
    subtitle = ", ".join(f"{c.label}: {len(c.records)} trials" for c in cohorts) if len(cohorts) > 1 \
        else f"{len(cohorts[0].records)} trials analyzed"
    rules = _rules_for(year_dim) + (_rules_for(series_dim) if series_dim else [])
    return BuildResult(_cartesian(VisualizationType.TIME_SERIES, title, subtitle, encoding, rows),
                       rules, warnings, time_granularity="year")


# ----------------------------------------------------------------------------- histogram

def _num(v: float) -> int | float:
    return int(v) if float(v).is_integer() else round(v, 4)


def _bin_label(lo: float, hi: float, integer: bool, thousands: bool) -> str:
    """'20–49' for integer data in [20, 50); '5,000+' for the open-ended last bin."""
    fmt = (lambda v: f"{v:,}") if thousands else str  # no "2,019" for years
    if math.isinf(hi):
        return f"{fmt(int(lo))}+" if integer else f"{lo:,.1f}+"
    if integer and float(lo).is_integer() and float(hi).is_integer() and hi - 1 >= lo:
        return f"{fmt(int(lo))}–{fmt(int(hi) - 1)}"
    return f"{lo:,.1f}–{hi:,.1f}"


def _trim_preset_edges(edges: list[float], top: float) -> list[float]:
    """Keep preset bins up to (and including) the one that contains the largest value."""
    for i in range(len(edges) - 1):
        if edges[i] <= top < edges[i + 1]:
            return edges[: i + 2]
    return edges


def build_histogram(cohorts: list[CohortData], field_key: NumericField, opts: BuildOptions) -> BuildResult:
    nd = NUMERIC_FIELDS[field_key]
    warnings: list[str] = []
    per_cohort = []
    all_values: list[float] = []
    for cohort in cohorts:
        points, missing = numeric_points(cohort.records, nd)
        prefix = f"{cohort.label}: " if len(cohorts) > 1 else ""
        warnings += [prefix + w for w in _missing_warning(missing, len(cohort.records), nd.title.lower())]
        per_cohort.append((cohort, points))
        all_values += [p.value for p in points]

    rows = []
    labels: list[str] = []
    if all_values:
        edges = (_trim_preset_edges(list(nd.bin_edges), max(all_values)) if nd.bin_edges
                 else equal_width_edges(all_values))
        integer = all(v.is_integer() for v in all_values)
        labels = [_bin_label(edges[i], edges[i + 1], integer, nd.thousands) for i in range(len(edges) - 1)]
        for cohort, points in per_cohort:
            for i, label in enumerate(labels):
                lo, hi = edges[i], edges[i + 1]
                last = i == len(edges) - 2
                bucket = Bucket(str(i), label)
                for p in points:
                    if lo <= p.value < hi or (last and p.value == hi):
                        bucket.add(p.record, p.evidence)
                fields = {"bin_label": label, "bin_start": _num(lo), "bin_end": None if math.isinf(hi) else _num(hi)}
                if len(cohorts) > 1:
                    fields["series"] = cohort.label
                rows.append(_row(bucket, opts.citation_limit, **fields))

    multi = len(cohorts) > 1
    encoding = CartesianEncoding(
        # Bins are drawn as ordered categories; bin_start/bin_end carry the numeric edges.
        x=Channel(field="bin_label", type="ordinal", title=nd.title, unit=nd.unit, sort=labels),
        y=COUNT_CHANNEL,
        series=Channel(field="series", type="nominal", title="Cohort") if multi else None,
        tooltip=["bin_label", COUNT_FIELD] + (["series"] if multi else []),
    )
    scope = " vs ".join(c.label for c in cohorts) if multi else describe_scope(cohorts[0].filters)
    rules = list(nd.notes) + ["Each bin covers [bin_start, bin_end); the last bin also includes its upper "
                              "edge, and bin_end = null means it is open-ended."]
    return BuildResult(
        _cartesian(VisualizationType.HISTOGRAM, f"Distribution of {nd.title.lower()}: {scope}",
                   f"{len(all_values)} trials with a value", encoding, rows),
        rules, warnings,
    )


# ----------------------------------------------------------------------------- scatter

def build_scatter(cohorts: list[CohortData], x_key: NumericField, y_key: NumericField,
                  opts: BuildOptions) -> BuildResult:
    xd, yd = NUMERIC_FIELDS[x_key], NUMERIC_FIELDS[y_key]
    rows, warnings = [], []
    for cohort in cohorts:
        dropped = 0
        for rec in cohort.records:
            xv, yv = xd.extract(rec), yd.extract(rec)
            if xv is None or yv is None:
                dropped += 1
                continue
            row = {"nct_id": rec.nct_id, "title": rec.title, xd.row_field: xv.value, yd.row_field: yv.value}
            if len(cohorts) > 1:
                row["series"] = cohort.label
            citations = [make_citation(rec, xv.evidence + yv.evidence)] if opts.citation_limit else []
            rows.append({**row, COUNT_FIELD: 1, "citation_count": 1, "citations": citations})
        if dropped:
            warnings.append(f"{dropped} trials lack {xd.title.lower()} or {yd.title.lower()} and are not plotted.")

    rows.sort(key=lambda r: (r[xd.row_field], r["nct_id"]))
    encoding = CartesianEncoding(
        x=Channel(field=xd.row_field, type="quantitative", title=xd.title, unit=xd.unit),
        y=Channel(field=yd.row_field, type="quantitative", title=yd.title, unit=yd.unit),
        series=Channel(field="series", type="nominal", title="Cohort") if len(cohorts) > 1 else None,
        tooltip=["nct_id", "title", xd.row_field, yd.row_field],
    )
    scope = " vs ".join(c.label for c in cohorts) if len(cohorts) > 1 else describe_scope(cohorts[0].filters)
    return BuildResult(
        _cartesian(VisualizationType.SCATTER_PLOT, f"{yd.title} vs {xd.title.lower()}: {scope}",
                   f"{len(rows)} trials (one point per trial)", encoding, rows),
        list(xd.notes + yd.notes) + ["Each point is one trial."], warnings,
    )


# ----------------------------------------------------------------------------- network

def build_network(cohorts: list[CohortData], spec: NetworkSpec, opts: BuildOptions) -> BuildResult:
    source = DIMENSIONS[ENTITY_DIMENSION[spec.source]]
    target = DIMENSIONS[ENTITY_DIMENSION[spec.target]]
    records = {r.nct_id: r for c in cohorts for r in c.records}.values()
    counts = co_occurrence(records, source, target, spec.source.value, spec.target.value, opts.ctx)

    min_weight = spec.min_edge_weight or 1
    strong = {k: b for k, b in counts.edges.items() if b.count >= min_weight}
    keep, edges = _select_subgraph(strong, counts.nodes, spec.top_n_nodes)
    connected = {n for pair in edges for n in pair}
    candidates = {n for pair in strong for n in pair}

    nodes = [
        NetworkNode(
            id=node_id, label=counts.nodes[node_id].label, entity_type=node_id.split(":", 1)[0],
            trial_count=counts.nodes[node_id].count, citation_count=counts.nodes[node_id].count,
            citations=counts.nodes[node_id].citations(opts.citation_limit),
        )
        for node_id in sorted(connected, key=lambda n: (-counts.nodes[n].count, n))
    ]
    edge_list = [
        NetworkEdge(source=a, target=b, trial_count=bucket.count, citation_count=bucket.count,
                    citations=bucket.citations(opts.citation_limit))
        for (a, b), bucket in sorted(edges.items(), key=lambda kv: (-kv[1].count, kv[0]))
    ]

    warnings = []
    if not edge_list:
        warnings.append(f"No {spec.source.value}–{spec.target.value} pairs appear in at least "
                        f"{min_weight} trials; try a broader query or a lower min_edge_weight.")
    if len(candidates) > len(connected):
        warnings.append(f"Showing {len(connected)} of {len(candidates)} connected entities "
                        f"(strongest links first, top_n_nodes={spec.top_n_nodes}).")

    same = spec.source == spec.target
    rules = _rules_for(source) + ([] if same else _rules_for(target)) + [
        (f"Edge weight = number of trials that include both {spec.source.value}s."
         if same else f"Edge weight = number of trials linking the {spec.source.value} and the {spec.target.value}."),
        "Node size (trial_count) = number of analyzed trials mentioning the entity, including trials "
        "whose edges were filtered out.",
        f"Edges supported by fewer than {min_weight} trials are hidden.",
    ]
    scope = " + ".join(c.label for c in cohorts) if len(cohorts) > 1 else describe_scope(cohorts[0].filters)
    title = (f"{spec.source.value.capitalize()} co-occurrence network: {scope}" if same
             else f"{spec.source.value.capitalize()}–{spec.target.value} network: {scope}")
    viz = NetworkVisualization(
        type=VisualizationType.NETWORK_GRAPH, title=title,
        subtitle=f"{len(nodes)} nodes, {len(edge_list)} edges from {len(records)} trials",
        data=NetworkData(nodes=nodes, edges=edge_list),
    )
    return BuildResult(viz, rules, warnings)


def _select_subgraph(strong: dict[tuple[str, str], Bucket], nodes: dict[str, Bucket],
                     budget: int) -> tuple[set[str], dict[tuple[str, str], Bucket]]:
    """Pick up to ``budget`` nodes by taking the strongest edges first.

    Ranking nodes independently (e.g. by degree) can select nodes that have no
    edges between them, or push out a hub whose links are individually weak but
    numerous. Growing the graph edge by edge keeps every kept node connected;
    ties go to edges between better-known entities, which favours hubs.
    """
    def edge_rank(item):
        (a, b), bucket = item
        return (-bucket.count, -(nodes[a].count + nodes[b].count), a, b)

    keep: set[str] = set()
    for (a, b), _ in sorted(strong.items(), key=edge_rank):
        new = {a, b} - keep
        if len(keep) + len(new) <= budget:
            keep |= new
        if len(keep) >= budget:
            break
    edges = {k: v for k, v in strong.items() if k[0] in keep and k[1] in keep}
    return keep, edges
