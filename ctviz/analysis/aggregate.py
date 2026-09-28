"""Counting primitives. Pure functions over TrialRecords; no I/O, no LLM.

Everything the service reports as a number is produced here, and every count is
the size of a set of NCT IDs we can cite.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations, product

from ..domain.records import TrialRecord
from ..schemas import Citation, Evidence
from .dimensions import MISSING, Category, DimensionDef, ExtractContext, NumericDef


@dataclass
class Bucket:
    """A set of trials sharing a category, with the evidence for each membership."""

    key: str
    label: str
    trials: dict[str, tuple[TrialRecord, tuple[Evidence, ...]]] = field(default_factory=dict)

    def add(self, record: TrialRecord, evidence: tuple[Evidence, ...]) -> None:
        # First evidence wins; a trial is counted at most once per bucket.
        self.trials.setdefault(record.nct_id, (record, evidence))

    @property
    def count(self) -> int:
        return len(self.trials)

    def citations(self, limit: int) -> list[Citation]:
        # Sorted by NCT ID so output is stable across runs and page orderings.
        chosen = sorted(self.trials)[:limit]
        return [make_citation(*self.trials[nct]) for nct in chosen]


def make_citation(record: TrialRecord, evidence: Iterable[Evidence]) -> Citation:
    combined: list[Evidence] = []
    for ev in [*evidence, *record.match_evidence]:
        if ev not in combined:
            combined.append(ev)
    return Citation(nct_id=record.nct_id, title=record.title, url=record.url, evidence=combined)


@dataclass
class Grouping:
    buckets: dict[str, Bucket]
    missing: int  # trials with no value for the dimension (and no missing bucket)


def group_by(records: Iterable[TrialRecord], dim: DimensionDef, ctx: ExtractContext) -> Grouping:
    buckets: dict[str, Bucket] = {}
    missing = 0
    for rec in records:
        categories = dim.extract(rec, ctx)
        if not categories:
            if dim.missing_label is None:
                missing += 1
                continue
            # Cite the absence itself: value None means "this field is not in the record".
            absent = (Evidence(field=dim.missing_path, value=None),) if dim.missing_path else ()
            categories = [Category(MISSING, dim.missing_label, absent)]
        for cat in categories:
            buckets.setdefault(cat.key, Bucket(cat.key, cat.label)).add(rec, cat.evidence)
    return Grouping(buckets, missing)


def ordered_buckets(grouping: Grouping, dim: DimensionDef) -> list[Bucket]:
    """Fixed order for ordinal dimensions, otherwise largest first (ties by label)."""
    buckets = list(grouping.buckets.values())
    if dim.order:
        rank = {k: i for i, k in enumerate(dim.order)}
        return sorted(buckets, key=lambda b: (rank.get(b.key, len(rank)), b.label))
    if dim.channel_type == "temporal":
        return sorted(buckets, key=lambda b: b.key)
    return sorted(buckets, key=lambda b: (-b.count, b.label.casefold()))


# ----------------------------------------------------------------------------- numeric

@dataclass
class NumericPoint:
    record: TrialRecord
    value: float
    evidence: tuple[Evidence, ...]


def numeric_points(records: Iterable[TrialRecord], nd: NumericDef) -> tuple[list[NumericPoint], int]:
    points, missing = [], 0
    for rec in records:
        v = nd.extract(rec)
        if v is None:
            missing += 1
        else:
            points.append(NumericPoint(rec, v.value, v.evidence))
    return points, missing


def equal_width_edges(values: list[float], max_bins: int = 20) -> list[float]:
    lo, hi = min(values), max(values)
    if lo == hi:
        return [lo, hi + 1]
    # Freedman–Diaconis is overkill here; sqrt rule bounded to a readable count.
    n_bins = max(1, min(max_bins, int(len(values) ** 0.5)))
    width = (hi - lo) / n_bins
    return [lo + i * width for i in range(n_bins)] + [hi]


# ----------------------------------------------------------------------------- co-occurrence

@dataclass
class NetworkCounts:
    nodes: dict[str, Bucket]              # node id -> trials mentioning the entity
    edges: dict[tuple[str, str], Bucket]  # (node id, node id) -> trials linking both


def co_occurrence(
    records: Iterable[TrialRecord],
    source: DimensionDef,
    target: DimensionDef,
    source_type: str,
    target_type: str,
    ctx: ExtractContext,
) -> NetworkCounts:
    """Count entity nodes and the trials that connect each pair of entities.

    Same entity type on both sides (drug<->drug) means undirected co-occurrence
    within a trial. Different types give a bipartite graph (sponsor<->drug).
    """
    nodes: dict[str, Bucket] = {}
    edges: dict[tuple[str, str], Bucket] = {}
    same_type = source.key == target.key

    for rec in records:
        left = [(f"{source_type}:{c.key}", c) for c in source.extract(rec, ctx)]
        right = left if same_type else [(f"{target_type}:{c.key}", c) for c in target.extract(rec, ctx)]
        for node_id, cat in left + ([] if same_type else right):
            nodes.setdefault(node_id, Bucket(node_id, cat.label)).add(rec, cat.evidence)

        pairs = combinations(sorted(left, key=lambda x: x[0]), 2) if same_type else product(left, right)
        for (a_id, a_cat), (b_id, b_cat) in pairs:
            if a_id == b_id:
                continue
            # combinations() over a sorted list already yields (smaller, larger) ids, and
            # bipartite edges keep source -> target orientation, so the key is canonical.
            key = (a_id, b_id)
            edges.setdefault(key, Bucket(f"{key[0]}|{key[1]}", "")).add(rec, a_cat.evidence + b_cat.evidence)

    return NetworkCounts(nodes, edges)
