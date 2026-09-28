"""Translate plan filters into ClinicalTrials.gov query parameters.

Every mapping below was checked against the live API (see "How it's designed" in the README):

* ``query.intr`` / ``query.cond`` are the registry's own search, which expands
  synonyms (MeSH terms, brand names). We use them for recall and verify drug
  matches afterwards (``domain.drugs.find_drug_evidence``, applied in pipeline.py).
* ``query.lead`` searches the lead sponsor only. ``query.spons`` also matches
  collaborators (Pfizer: 6,087 vs 3,880 trials), which is not what people mean
  by "Pfizer's trials".
* Phase, country, study type and start-date limits go through
  ``filter.advanced`` with the Essie ``AREA[...]`` syntax.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..schemas import PlanFilters


def _quote(value: str) -> str:
    # Essie treats quoted text as a phrase; strip quotes the user may have typed.
    return '"' + value.replace('"', "").strip() + '"'


def build_search_params(filters: PlanFilters, fields: Iterable[str]) -> dict[str, str]:
    params: dict[str, str] = {}

    if filters.drug:
        params["query.intr"] = filters.drug
    if filters.condition:
        params["query.cond"] = filters.condition
    if filters.sponsor:
        params["query.lead"] = filters.sponsor
    if filters.statuses:
        params["filter.overallStatus"] = ",".join(s.value for s in filters.statuses)

    clauses: list[str] = []
    if filters.phases:
        joined = " OR ".join(p.value for p in filters.phases)
        clauses.append(f"AREA[Phase]({joined})" if len(filters.phases) > 1 else f"AREA[Phase]{joined}")
    if filters.country:
        clauses.append(f"AREA[LocationCountry]{_quote(filters.country)}")
    if filters.study_type:
        clauses.append(f"AREA[StudyType]{filters.study_type.value}")
    if filters.start_year or filters.end_year:
        lo = f"{filters.start_year}-01-01" if filters.start_year else "MIN"
        hi = f"{filters.end_year}-12-31" if filters.end_year else "MAX"
        clauses.append(f"AREA[StartDate]RANGE[{lo},{hi}]")
    if clauses:
        params["filter.advanced"] = " AND ".join(clauses)

    params["fields"] = ",".join(sorted(set(fields)))
    return params
