"""Keyword-based planner used when the LLM is unavailable or fails.

It is intentionally conservative: it recognises the question shapes listed in
the assignment (trends, distributions, comparisons, geography, networks,
histograms, scatter plots) and pulls out years, phases and statuses reliably.
Entity extraction from free text is heuristic, which is why explicit request
fields (drug_name, condition, ...) always take precedence over it.
"""

from __future__ import annotations

import re

from ..domain.drugs import SYNONYMS
from ..schemas import (
    AnalysisType,
    CompareField,
    CompareSpec,
    Dimension,
    EntityType,
    NetworkSpec,
    NumericField,
    NumericSpec,
    OverallStatus,
    Phase,
    PlanFilters,
    QueryPlan,
)

# International nonproprietary name stems: a cheap but decent "is this a drug?" test.
_DRUG_STEMS = (
    "mab", "nib", "tide", "parib", "lisib", "ciclib", "platin", "taxel", "rubicin", "stat", "pril",
    "sartan", "gliflozin", "gliptin", "azole", "vir", "mide", "tinib", "zumab", "ximab", "cept",
    "cycline", "mycin", "olol", "prazole", "semide", "dronate", "lukast", "vastatin", "formin",
    "rafenib", "degib", "tecan", "cabtagene", "leucel", "vedotin", "deruxtecan", "tuximab",
)
_KNOWN_DRUGS = set(SYNONYMS) | set(SYNONYMS.values())

_STATUS_PATTERNS = [
    (r"\bnot yet recruiting\b", OverallStatus.NOT_YET_RECRUITING),
    (r"(?<!not yet )\brecruiting\b|\bopen trials\b|\benrolling\b", OverallStatus.RECRUITING),
    (r"\bcompleted\b", OverallStatus.COMPLETED),
    (r"\bterminated\b", OverallStatus.TERMINATED),
    (r"\bwithdrawn\b", OverallStatus.WITHDRAWN),
    (r"\bactive,? not recruiting\b", OverallStatus.ACTIVE_NOT_RECRUITING),
]

_COMPARE_SEPARATOR = re.compile(
    r"\s*,\s*(?:and\s+)?|\s+(?:vs\.?|versus|and|compared (?:to|with)|with)\s+", re.I
)

_PHASE_WORDS = {"1": Phase.PHASE1, "i": Phase.PHASE1, "2": Phase.PHASE2, "ii": Phase.PHASE2,
                "3": Phase.PHASE3, "iii": Phase.PHASE3, "4": Phase.PHASE4, "iv": Phase.PHASE4}

_GROUP_PATTERNS: list[tuple[str, Dimension]] = [
    (r"\bsponsor (?:type|types|categor\w*|class\w*)\b|\bindustry vs\.? academic\b", Dimension.SPONSOR_CLASS),
    (r"\bintervention types?\b|\btypes? of intervention", Dimension.INTERVENTION_TYPE),
    (r"\bstudy types?\b|\bobservational\b.*\binterventional\b", Dimension.STUDY_TYPE),
    (r"\bphases?\b", Dimension.PHASE),
    (r"\bcountr(?:y|ies)\b|\bgeograph\w*|\bwhere\b|\bregions?\b", Dimension.COUNTRY),
    (r"\bsponsors?\b|\bcompanies\b|\borgani[sz]ations?\b", Dimension.LEAD_SPONSOR),
    (r"\bstatus(?:es)?\b", Dimension.STATUS),
    (r"\bconditions\b|\bdiseases\b|\bindications\b", Dimension.CONDITION),
    (r"\bdrugs\b|\btreatments\b|\btherapies\b|\binterventions\b", Dimension.DRUG),
]

_ENTITY_WORDS = [
    (r"\bdrugs?\b|\btreatments?\b|\btherapies\b|\binterventions?\b", EntityType.DRUG),
    (r"\bsponsors?\b|\bcompanies\b|\borgani[sz]ations?\b", EntityType.SPONSOR),
    (r"\bconditions?\b|\bdiseases?\b|\bindications?\b", EntityType.CONDITION),
    (r"\bcountr(?:y|ies)\b", EntityType.COUNTRY),
]

_STOPWORDS = {
    "the", "a", "an", "all", "this", "that", "these", "those", "trial", "trials", "study", "studies",
    "drug", "drugs", "clinical", "recruiting", "completed", "phase", "phases", "number", "each", "year",
}

# Words that describe the *question* rather than the subject. What is left after
# removing them (and numbers) is usually the drug or condition being asked about.
_QUESTION_WORDS = _STOPWORDS | {
    "how", "many", "much", "what", "which", "who", "where", "when", "is", "are", "was", "were", "do",
    "does", "did", "has", "have", "had", "been", "be", "can", "show", "me", "give", "list", "plot",
    "chart", "visualize", "visualise", "graph", "display", "draw", "compare", "comparison", "of", "for",
    "in", "on", "by", "per", "to", "from", "since", "after", "before", "until", "between", "across",
    "over", "with", "and", "or", "vs", "versus", "against", "involving", "testing", "studying", "their",
    "its", "most", "common", "frequent", "frequently", "top", "largest", "count", "counts", "changed",
    "change", "changes", "trend", "trends", "time", "years", "yearly", "annual", "annually", "started",
    "start", "starting", "distributed", "distribution", "breakdown", "split", "types", "type",
    "intervention", "interventions", "countries", "country", "sponsor", "sponsors", "network",
    "co-occur", "cooccur", "occur", "combination", "combinations", "enrollment", "enrolment", "size",
    "sizes", "sample", "duration", "length", "long", "large", "categories", "category", "class",
    "status", "statuses", "conditions", "diseases", "treatments", "therapies", "companies", "open",
    "active", "terminated", "withdrawn", "not", "yet", "i", "ii", "iii", "iv", "relationship", "links",
    "scatter", "histogram", "timeline", "registered", "ongoing", "new", "there", "being",
}


def looks_like_drug(term: str) -> bool:
    t = term.lower().strip()
    return t in _KNOWN_DRUGS or any(t.endswith(stem) for stem in _DRUG_STEMS) or bool(
        re.fullmatch(r"[a-z]{1,5}-?\d{2,}[a-z0-9-]*", t)  # development codes like mk-3475, ly3298176
    )


def _clean_entity(text: str) -> str | None:
    words = [w for w in re.split(r"\s+", text.strip(" .,?!'\"")) if w]
    while words and words[0].lower() in _STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in _STOPWORDS:
        words.pop()
    cleaned = " ".join(words).strip(" .,?!'\"")
    return cleaned or None


class RuleBasedPlanner:
    method = "rule_based"

    def plan(self, query: str, explicit_drug: str | None = None) -> QueryPlan:
        """``explicit_drug`` resolves "this drug" in comparisons; other explicit fields are
        merged later by resolve_plan."""
        q = " ".join(query.lower().split())
        filters = PlanFilters()
        self._years(q, filters)
        self._statuses(q, filters)
        self._phases(q, filters)

        compare = self._compare(query, explicit_drug)
        analysis = self._analysis(q, compare)
        entity = None if compare else self._scope_entity(query)
        if entity and looks_like_drug(entity):
            filters.drug = entity
        elif entity:
            filters.condition = entity

        plan = QueryPlan(analysis=analysis, filters=filters, compare=compare,
                         rationale="Rule-based interpretation (LLM planner not used).")

        if analysis == AnalysisType.NETWORK:
            plan.network = self._network(q, filters)
        elif analysis == AnalysisType.HISTOGRAM:
            plan.numeric = NumericSpec(x=self._numeric(q) or NumericField.ENROLLMENT)
        elif analysis == AnalysisType.SCATTER:
            fields = self._numeric_all(q)
            plan.numeric = NumericSpec(x=fields[0] if fields else NumericField.START_YEAR,
                                       y=fields[1] if len(fields) > 1 else NumericField.ENROLLMENT)
        else:
            trend = analysis == AnalysisType.TREND
            grouping = self._group_by(q, compare, trend=trend)
            if trend:
                plan.group_by, plan.series_by = Dimension.START_YEAR, grouping
            else:
                plan.group_by = grouping or Dimension.PHASE
        return plan

    # ---------------------------------------------------------------- pieces

    @staticmethod
    def _years(q: str, f: PlanFilters) -> None:
        if (m := re.search(r"\b(?:between|from) (\d{4}) (?:and|to|-|until) (\d{4})\b", q)) or (m := re.search(r"\b(\d{4})\s*[-–]\s*(\d{4})\b", q)):
            f.start_year, f.end_year = int(m.group(1)), int(m.group(2))
        else:
            if m := re.search(r"\b(?:since|from|after|starting(?: in)?) (\d{4})\b", q):
                f.start_year = int(m.group(1)) + (1 if "after" in m.group(0) else 0)
            if m := re.search(r"\b(?:before|until|through|up to) (\d{4})\b", q):
                f.end_year = int(m.group(1)) - (1 if "before" in m.group(0) else 0)
            if not (f.start_year or f.end_year) and (m := re.search(r"\bin (\d{4})\b", q)):
                f.start_year = f.end_year = int(m.group(1))

    @staticmethod
    def _statuses(q: str, f: PlanFilters) -> None:
        for pattern, status in _STATUS_PATTERNS:
            if re.search(pattern, q) and status not in f.statuses:
                f.statuses.append(status)
        # "active, not recruiting" also matches the bare "recruiting" rule
        if OverallStatus.ACTIVE_NOT_RECRUITING in f.statuses and OverallStatus.RECRUITING in f.statuses \
                and not re.search(r"(?<!not )\brecruiting\b(?!.*not recruiting)", q):
            f.statuses.remove(OverallStatus.RECRUITING)

    @staticmethod
    def _phases(q: str, f: PlanFilters) -> None:
        # Only explicit "phase 3"-style mentions become filters; "by phase" is a grouping.
        # "phase 1/2" becomes both phases (the API filter is an OR).
        for m in re.finditer(r"\bphase[\s-]?(1|2|3|4|iv|iii|ii|i)(?:\s*/\s*(?:phase\s*)?(2|3|ii|iii))?\b", q):
            for code in filter(None, m.groups()):
                phase = _PHASE_WORDS[code]
                if phase not in f.phases:
                    f.phases.append(phase)

    @staticmethod
    def _compare(query: str, explicit_drug: str | None = None) -> CompareSpec | None:
        """'semaglutide vs tirzepatide', 'across lung cancer and melanoma', 'A vs B vs C',
        'compare this drug with nivolumab' (with drug_name supplied)."""
        text = query.strip().rstrip("?.!")
        if not re.search(r"\bcompar\w*|\bvs\.?(?=\s)|\bversus\b", text, re.I):
            return None
        m = re.search(r"\b(?:involving|for|of|between|across|in)\s+(.+?)(?:\s+(?:trials?|studies)\b|$)", text, re.I)
        span = m.group(1) if m and _COMPARE_SEPARATOR.search(m.group(1)) else None
        if span is None:
            span = re.sub(r"^\s*compare\s+", "", text, flags=re.I)
            span = re.sub(r"\s+(?:trials?|studies)\b.*$", "", span, flags=re.I)

        values: list[str] = []
        for part in _COMPARE_SEPARATOR.split(span):
            if re.search(r"\bthis (?:drug|medication|treatment)\b", part, re.I):
                if explicit_drug:
                    values.append(explicit_drug)
                continue
            # "phases for trials involving semaglutide" -> "semaglutide"; "opdivo by phase" -> "opdivo"
            part = re.split(r"\b(?:involving|for|of|on|across|between)\s+", part, flags=re.I)[-1]
            part = re.split(r"\s+(?:by|per|over|since|from|in)\s+", part, flags=re.I)[0]
            value = _clean_entity(part)
            if value and value.lower() not in {v.lower() for v in values}:
                values.append(value)
        if len(values) < 2:
            return None
        values = values[:5]
        is_drug = bool(explicit_drug and explicit_drug in values) or any(looks_like_drug(v) for v in values)
        return CompareSpec(field=CompareField.DRUG if is_drug else CompareField.CONDITION, values=values)

    @staticmethod
    def _analysis(q: str, compare: CompareSpec | None) -> AnalysisType:
        if re.search(r"\bnetwork\b|\bco-?occur\w*|\bgraph of\b|\bconnect\w*\b|\blinks? between\b", q):
            return AnalysisType.NETWORK
        if re.search(r"\bscatter\b|\bcorrelat\w*|\bagainst\b|\brelationship between\b", q):
            return AnalysisType.SCATTER
        if re.search(r"\bhistogram\b|\bdistribution of (?:enrollment|sample size|trial size|duration)|"
                     r"\benrollment sizes?\b|\bhow (?:large|big|long)\b|\bsample sizes?\b", q):
            return AnalysisType.HISTOGRAM
        if re.search(r"\bover time\b|\bper year\b|\beach year\b|\bby year\b|\byearly\b|\bannual\w*\b|"
                     r"\btrend\w*\b|\btimeline\b|\bchanged\b|\bgrowth\b", q):
            return AnalysisType.TREND
        if compare:
            return AnalysisType.COMPARISON
        return AnalysisType.DISTRIBUTION

    @staticmethod
    def _group_by(q: str, compare: CompareSpec | None, trend: bool = False) -> Dimension | None:
        if trend:
            # In a trend the x-axis is time; a second grouping must be asked for explicitly
            # ("... per year by phase"), otherwise "for Alzheimer's disease" would become one.
            m = re.search(r"\b(?:by|for each|split by|broken down by)\s+(?!year)(.*)$", q)
            if not m:
                return None
            q = m.group(1)
        for pattern, dim in _GROUP_PATTERNS:
            if not re.search(pattern, q):
                continue
            if compare and dim.value == compare.field.value:
                continue
            # "phase 3 trials" is a filter, not a grouping
            if dim == Dimension.PHASE and not trend and \
                    not re.search(r"\bphases\b|\b(?:by|across|per) phase\b", q):
                continue
            return dim
        return None

    @staticmethod
    def _scope_entity(query: str) -> str | None:
        """The subject of the question: the longest run of words that aren't question vocabulary.

        A run introduced by a preposition ("for melanoma", "involving nivolumab") is
        preferred, since that is how people name the subject.
        """
        tokens = re.findall(r"[A-Za-z0-9][\w\-']*", query)
        runs: list[tuple[bool, list[str]]] = []
        current: list[str] = []
        after_prep = False
        prev = ""
        for tok in tokens:
            low = tok.lower()
            if low in _QUESTION_WORDS or re.fullmatch(r"\d+", low):
                if current:
                    runs.append((after_prep, current))
                    current = []
            else:
                if not current:
                    after_prep = prev in {"for", "involving", "of", "in", "on", "testing", "studying"}
                current.append(tok)
            prev = low
        if current:
            runs.append((after_prep, current))
        if not runs:
            return None
        best = max(runs, key=lambda r: (r[0], len(r[1])))
        return " ".join(best[1])

    @staticmethod
    def _network(q: str, f: PlanFilters) -> NetworkSpec:
        mentioned = []
        for pattern, entity in _ENTITY_WORDS:
            if (m := re.search(pattern, q)):
                mentioned.append((m.start(), entity))
        mentioned.sort()
        types = [e for _, e in mentioned]
        if len(types) >= 2 and types[0] != types[1]:
            return NetworkSpec(source=types[0], target=types[1])
        if types == [EntityType.DRUG] or re.search(r"\bco-?occur|\bcombination", q):
            return NetworkSpec(source=EntityType.DRUG, target=EntityType.DRUG)
        if f.condition:
            return NetworkSpec(source=EntityType.SPONSOR, target=EntityType.DRUG)
        return NetworkSpec(source=EntityType.DRUG, target=EntityType.DRUG)

    @staticmethod
    def _numeric_all(q: str) -> list[NumericField]:
        found = []
        for pattern, fieldname in [
            (r"\benrol\w*|\bsample size|\bparticipants|\btrial size", NumericField.ENROLLMENT),
            (r"\bduration\b|\bhow long\b|\blength\b", NumericField.DURATION_MONTHS),
            (r"\bstart (?:year|date)\b|\byear\b", NumericField.START_YEAR),
        ]:
            if (m := re.search(pattern, q)):
                found.append((m.start(), fieldname))
        return [f for _, f in sorted(found)]

    def _numeric(self, q: str) -> NumericField | None:
        fields = self._numeric_all(q)
        return fields[0] if fields else None
