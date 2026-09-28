"""Intervention name cleanup and drug matching.

Registry intervention names are free text. For pembrolizumab alone we observed
"Pembrolizumab", "MK-3475", "Pembrolizumab Injection [Keytruda]",
"KEYTRUDA ®( Pembrolizumab)", "Pembrolizumab 200 mg", "lenvatinib plus pembrolizumab"
and "Placebo for pembrolizumab". Two jobs follow from that:

1. ``canonical_drugs`` turns one intervention into zero or more canonical drug
   names (split combinations, drop dosages/descriptors/placebos, map brand and
   code names to the generic). Networks and drug grouping use this.
2. ``find_drug_evidence`` decides whether a trial really involves a drug, and
   returns the exact field that says so. ``query.intr`` matched ~16% of
   pembrolizumab results that never name it in an intervention (it searches
   descriptions and keywords too), so we verify before counting.

The synonym table is intentionally small and easy to extend. With more time it
should be replaced by RxNorm/UNII lookups.
"""

from __future__ import annotations

import re

from .records import Intervention, Sourced, TrialRecord

DRUG_LIKE_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}

# brand / development code -> generic name (all lower case)
SYNONYMS: dict[str, str] = {
    "keytruda": "pembrolizumab", "mk-3475": "pembrolizumab", "mk3475": "pembrolizumab",
    "lambrolizumab": "pembrolizumab", "sch 900475": "pembrolizumab",
    "opdivo": "nivolumab", "bms-936558": "nivolumab", "mdx-1106": "nivolumab", "ono-4538": "nivolumab",
    "tecentriq": "atezolizumab", "mpdl3280a": "atezolizumab", "rg7446": "atezolizumab",
    "imfinzi": "durvalumab", "medi4736": "durvalumab",
    "yervoy": "ipilimumab", "mdx-010": "ipilimumab", "bms-734016": "ipilimumab",
    "libtayo": "cemiplimab", "regn2810": "cemiplimab",
    "bavencio": "avelumab", "msb0010718c": "avelumab",
    "herceptin": "trastuzumab", "avastin": "bevacizumab", "erbitux": "cetuximab",
    "padcev": "enfortumab vedotin", "lenvima": "lenvatinib", "tagrisso": "osimertinib",
    "abraxane": "nab-paclitaxel", "taxol": "paclitaxel", "taxotere": "docetaxel",
    "temodar": "temozolomide", "temodal": "temozolomide",
    "ozempic": "semaglutide", "wegovy": "semaglutide", "rybelsus": "semaglutide",
    "mounjaro": "tirzepatide", "zepbound": "tirzepatide", "ly3298176": "tirzepatide",
    "trulicity": "dulaglutide", "victoza": "liraglutide", "saxenda": "liraglutide",
    "jardiance": "empagliflozin", "farxiga": "dapagliflozin", "forxiga": "dapagliflozin",
    "glucophage": "metformin", "humira": "adalimumab", "leqembi": "lecanemab",
    "kisunla": "donanemab", "aduhelm": "aducanumab",
}

# Interventions that are not a specific drug. Matched as whole words.
NON_DRUG_TERMS = {
    "placebo", "saline", "vehicle", "sham", "standard of care", "soc", "usual care",
    "best supportive care", "observation", "control", "no intervention", "chemotherapy",
    "physician's choice", "investigator's choice", "matching placebo",
}

_DESCRIPTOR_WORDS = (
    "injection", "infusion", "intravenous(?:ly)?", "iv", "subcutaneous(?:ly)?", "sc", "oral(?:ly)?",
    "tablets?", "intra-?arterial(?:ly)?", "intrathecal(?:ly)?", "intratumou?ral(?:ly)?", "intranasal",
    "topical", "administered", "given",
    "capsules?", "solution", "monotherapy", "neoadjuvant", "adjuvant", "maintenance",
    "induction", "consolidation", "dose escalation", "dose expansion", "low dose", "high dose",
    "arm [a-z0-9]+", "cohort [a-z0-9]+", "regimen", "treatment", "co-formulation",
)
_DESCRIPTOR_RE = re.compile(r"\b(?:" + "|".join(_DESCRIPTOR_WORDS) + r")\b")
_DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg/kg|mg/m2|mg/m²|mg|mcg|µg|ug|g|ml|iu|units?|%)(?:/\w+)?\b"
    r"|\bauc\s*\d+(?:\.\d+)?\b"   # carboplatin dosing
    r"|\bq\d+w\b"                    # schedules like q3w
)
_ROUTE_ABBREV_RE = re.compile(r"\b(?:s\.c|i\.v|p\.o|i\.m)\.?(?=\s|$)")
_BRACKETS_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_LABEL_PREFIX_RE = re.compile(r"^[\w\s\-]{1,40}:\s*")  # "Combination Therapy: X + Y"
_SPLIT_RE = re.compile(r"\s*(?:\+|/|,|;|&|\bplus\b|\band\b|\bwith\b|\bin combination with\b)\s*")
_WS_RE = re.compile(r"\s+")


def _basic_clean(text: str) -> str:
    text = text.lower().replace("®", " ").replace("™", " ").replace("©", " ")
    return _WS_RE.sub(" ", text).strip()


def _clean_piece(piece: str) -> str:
    piece = _DOSE_RE.sub(" ", piece)
    piece = _DESCRIPTOR_RE.sub(" ", piece)
    piece = re.sub(r"[^\w\s\-]", " ", piece)
    piece = _WS_RE.sub(" ", piece).strip(" -")
    return SYNONYMS.get(piece, piece)


def _is_non_drug(piece: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", piece) for term in NON_DRUG_TERMS)


def canonical_drug_names(raw_name: str) -> list[str]:
    """'Pembrolizumab Injection [Keytruda] + Lenvatinib 20 mg' -> ['pembrolizumab', 'lenvatinib']."""
    text = _basic_clean(raw_name).replace("(+)", " + ")
    text = _ROUTE_ABBREV_RE.sub(" ", text)
    if _is_non_drug(text) and not _SPLIT_RE.search(text):
        return []

    # Content in brackets is usually a synonym of what precedes it ("X (MK-3475)");
    # but when the name *starts* with a brand and the generic is in brackets
    # ("KEYTRUDA ( Pembrolizumab)") the synonym table maps the brand anyway.
    text = _BRACKETS_RE.sub(" ", text)
    text = _LABEL_PREFIX_RE.sub("", text)
    text = _DOSE_RE.sub(" ", text)  # before splitting so "mg/kg" isn't split on "/"

    names: list[str] = []
    for piece in _SPLIT_RE.split(text):
        piece = piece.strip()
        if not piece or _is_non_drug(piece):
            continue
        cleaned = _clean_piece(piece)
        if len(cleaned) < 3 or cleaned.isdigit():
            continue
        if cleaned not in names:
            names.append(cleaned)
    return names


def canonical_drugs(intervention: Intervention) -> list[str]:
    """Canonical drug names for a drug-like intervention; [] for procedures, devices, etc."""
    if intervention.type and intervention.type not in DRUG_LIKE_TYPES:
        return []
    names = canonical_drug_names(intervention.name.raw)
    # "MK-3475" style names that we don't know: use the registry's own otherNames
    # if one of them is a generic name we do know.
    known_generics = set(SYNONYMS.values())
    resolved = []
    for name in names:
        if name not in known_generics:
            for other in intervention.other_names:
                mapped = [n for n in canonical_drug_names(other.raw) if n in known_generics]
                if mapped:
                    name = mapped[0]
                    break
        if name not in resolved:
            resolved.append(name)
    return resolved


def display_name(canonical: str) -> str:
    return " ".join(w if any(c.isdigit() for c in w) else w.capitalize() for w in canonical.split(" "))


def drug_aliases(term: str) -> set[str]:
    """All spellings we accept for a drug the user asked about."""
    canonical = SYNONYMS.get(_basic_clean(term), _basic_clean(term))
    aliases = {canonical} | {alias for alias, generic in SYNONYMS.items() if generic == canonical}
    return {a for a in aliases if a}


def _mentions(text: str, aliases: set[str]) -> bool:
    cleaned = _basic_clean(text)
    # Hyphens count as boundaries ("LSG-Tirzepatide group"), letters and digits don't.
    return any(re.search(rf"(?<![^\W_]){re.escape(a)}(?![^\W_])", cleaned) for a in aliases)


def find_drug_evidence(record: TrialRecord, term: str) -> Sourced[str] | None:
    """Return the field proving the trial involves ``term``, or None.

    Checked in order of strength: intervention name, the intervention's registered
    other names, then the brief title (some trials, mostly observational, list no
    interventions at all). Placebo arms ("Placebo for pembrolizumab") are not evidence.
    """
    aliases = drug_aliases(term)
    for intervention in record.interventions:
        name = intervention.name.raw
        if _mentions(name, aliases) and not re.search(r"\bplacebo\b", name, re.I):
            return intervention.name
    for intervention in record.interventions:
        if re.search(r"\bplacebo\b", intervention.name.raw, re.I):
            continue
        for other in intervention.other_names:
            if _mentions(other.raw, aliases):
                return other
    if record.title and _mentions(record.title, aliases):
        return Sourced(record.title, "protocolSection.identificationModule.briefTitle", record.title)
    return None
