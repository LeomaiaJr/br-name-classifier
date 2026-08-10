"""Precision-first policy layered over the base BR name classifier.

The underlying model is intentionally broad: it detects names that resemble
its Lusophone training population.  It is a closed-set model trained mainly
against Hispanic and US-English negatives, so out-of-distribution names can
receive extreme scores.  This module makes High and Very High application
claims conservative while preserving the broad base model for research use.

Policy:

* High requires corroborated Lusophone evidence or credible Portuguese name
  structure.  An ordinary surname match by itself remains Medium.
* Very High also requires independent corroboration (a frequency-backed
  given-name signal, another surname, or a compound given name).  Structural
  rescues are deliberately High-only.
* Strong US Census cross-cultural surname aggregates cap unsupported records
  at Low.  They are conflict signals only, never identity attribution.
* Every owner and every explicit person segment is qualified independently.
  A property record uses the strongest qualified owner, so evidence is never
  borrowed across owners and two weak names cannot combine into High.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CLASSIFIER_DIR = ROOT

import scorer  # noqa: E402
from constants import (  # noqa: E402
    ALL_PT_PREPS,
    BR_SUFFIXES,
    BUSINESS_KEYWORDS,
    COMPOUND_FIRST_NAMES,
    CURATED_BR_COMPOUND_GIVEN_NAMES,
    CURATED_AO_TOKENS,
    CURATED_CV_SURNAMES,
    CURATED_MZ_SURNAMES,
    MERGED_PREPS,
    SUFFIX_STRIP,
)


POLICY_VERSION = "2026-08-05"
CONFLICT_LOOKUP_PATH = ROOT / "output/surname_conflicts.json"
LUSOPHONE_CLASSES = ("br", "pt", "cv", "ao", "mz", "palop_other")
POLICY_CONNECTORS = frozenset((*ALL_PT_PREPS, "E"))
VALID_COMPOUND_SURNAMES = frozenset(
    {
        "DA SILVA",
        "DE SOUZA",
        "DE SOUSA",
        "DE OLIVEIRA",
        "DE ALMEIDA",
        "DE JESUS",
        "DOS SANTOS",
        "DO CARMO",
        "DE PAULA",
        "DE FREITAS",
        "DE CASTRO",
        "DE LIMA",
        "DE MOURA",
        "DE BRITO",
        "DE ARAUJO",
        "DA COSTA",
        "DA ROCHA",
        "DA CRUZ",
        "DA CUNHA",
        "DA LUZ",
        "DE PINA",
    }
)
VALID_COMPOUND_SEQUENCES = tuple(
    tuple(name.split()) for name in VALID_COMPOUND_SURNAMES
)
POLICY_ORGANIZATION_KEYWORDS = frozenset(
    {
        "ACADEMY",
        "APARTMENT",
        "APARTMENTS",
        "ASSOC",
        "ASSOCIATES",
        "BAPTIST",
        "BRIDGE",
        "BUILDERS",
        "BUILDING",
        "CENTER",
        "CLINICAL",
        "CLUB",
        "CLEANING",
        "COLLEGE",
        "COMMERCIAL",
        "CONDOMIN",
        "CONSTRUCTION",
        "DEVELOPMENT",
        "ENTERPRISE",
        "FINANCIAL",
        "FOUNDATION",
        "FUND",
        "GARAGE",
        "GROUP",
        "HOMES",
        "HOMEOWNERS",
        "HOSPITAL",
        "INVERSIONES",
        "LIMITED",
        "LLLP",
        "PAVERS",
        "POA",
        "PLLC",
        "PROFESSIONAL",
        "RANCH",
        "REALTY",
        "RENTAL",
        "RESTAURANT",
        "RESOURCES",
        "SCHOOL",
        "SERVICE",
        "SERVICES",
        "SQUARE",
        "STREET",
        "TIRE",
        "TOWNHOME",
        "TOWNHOMES",
        "TOWNHOUSE",
        "TRUCKING",
        "UNIVERSITY",
        "UTILITIES",
        "UTILITY",
        "AT",
    }
)
POLICY_ORGANIZATION_PHRASES = (
    "HOME SERVICE",
    "LAND TRUST",
    "LAND TR",
    "OF NORTH FLORIDA",
    "REAL ESTATE",
    "REAL ESTAT",
    "REALTY TRUST",
    "SANTA ROSA BAY",
    "SANTA ROSA BEACH",
)
POLICY_ORGANIZATION_STEMS = (
    "APART",
    "ASSOC",
    "COMMERC",
    "CONDOM",
    "CONSTR",
    "DEVELOP",
    "INVEST",
    "MORTG",
    "PARTICIP",
    "PROPERT",
    "SERVIC",
    "TOWNH",
)
HISPANIC_STRUCTURE_PHRASES = ("DE LOS", "DE LAS", "DE LA")
MERGED_HISPANIC_STRUCTURES = frozenset({"DELA", "DELAS", "DELOS"})
AMBIGUOUS_BARE_SURNAMES = frozenset({"PAULA", "SA"})
AMBIGUOUS_COMPOUND_SURNAMES = (("DE", "SILVA"),)
COLLISION_MERGED_SURNAMES = frozenset({"DACOSTA", "DECASTRO", "DEJESUS"})
COUNTRY_CLUSTER_SURNAMES = frozenset(
    (*CURATED_CV_SURNAMES, *CURATED_MZ_SURNAMES, *CURATED_AO_TOKENS)
)
# Curated diaspora lists must not override overwhelming contrary evidence in
# the broader surname tables. These two entries are overwhelmingly Hispanic
# in the available counts despite sparse Cape Verdean examples.
COUNTRY_CLUSTER_COLLISIONS = frozenset({"EVORA", "VARELA"})
TRUNCATED_PROPERTY_TOKEN_PREFIXES = frozenset({"MAR", "MARTI", "PERE"})
HIGH_CONFIDENCE_HISPANIC_GIVEN_OVERRIDES = frozenset({"JOAQUIN"})
COMPOUND_SEQUENCES = tuple(tuple(name.split()) for name in COMPOUND_FIRST_NAMES)
BR_COMPOUND_SEQUENCES = tuple(
    tuple(name.split()) for name in CURATED_BR_COMPOUND_GIVEN_NAMES
)
STRONG_PORTUGUESE_SUFFIXES = frozenset({"FILHO", "NETO", "SOBRINHO"})
PROPERTY_METADATA_SUFFIXES = tuple(
    sorted(
        SUFFIX_STRIP,
        key=lambda suffix: (len(suffix.split()), len(suffix)),
        reverse=True,
    )
)
POLICY_SHARED_IBERIAN_SURNAMES = frozenset(
    {
        "ALVES",
        "ANDRADE",
        "BATISTA",
        "CHAVES",
        "FERNANDES",
        "FONSECA",
        "GOMES",
        "GONCALVES",
        "LOPES",
        "MARQUES",
        "MENDES",
        "NUNES",
        "PIRES",
        "RODRIGUES",
        "SOARES",
    }
)
OWNER_SEGMENT_RE = re.compile(r"\s*(?:&|\+|\bAND\b)\s*")


@dataclass(frozen=True)
class PrecisionEvidence:
    tier: int
    census_conflict: str | None
    organization: bool
    surname_tokens: tuple[str, ...]
    guarded_rescue: bool = False
    high_only: bool = False
    floor_eligible: bool = False
    natural_order_rescue: bool = False
    supporting_given_rescue: bool = False
    api_medium_rescue: bool = False
    country_hispanic_surname_conflict: bool = False
    substantive_content_count: int = 0
    strong_portuguese_suffix_present: bool = False


@lru_cache(maxsize=1)
def _policy_tables() -> tuple[dict, dict, dict[str, dict[str, float | int]]]:
    with (CLASSIFIER_DIR / "output/frequency_tables.json").open(encoding="utf-8") as handle:
        frequencies = json.load(handle)
    with (CLASSIFIER_DIR / "output/country_tables.json").open(encoding="utf-8") as handle:
        countries = json.load(handle)
    with CONFLICT_LOOKUP_PATH.open(encoding="utf-8") as handle:
        conflicts = json.load(handle).get("data", {})
    return frequencies, countries, conflicts


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_name(value: str) -> tuple[str, bool]:
    if not value:
        return "", False
    name = _strip_accents(value).upper().strip().replace(",", " ").replace("-", " ")
    name = re.sub(r"\bL\s*\.?\s*L\s*\.?\s*C\b\.?", "LLC", name)
    name = re.sub(r"^REM:\s*", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name.startswith("THE "):
        name = name[4:].strip()
    pre_suffix_name = name
    for suffix in SUFFIX_STRIP:
        if name.endswith(" " + suffix):
            name = name[: -(len(suffix) + 1)]
    # LE H/E (and the H or E variants) is unambiguous life-estate metadata.
    # Bare terminal LE is handled later as a High-only exception because LE
    # can also be a real surname in a generic name checker.
    name = re.sub(r"\s+LE\s+(?:H(?:/E)?|E)$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    post_suffix_words = set(re.findall(r"[A-Z]+", name))
    pre_suffix_words = set(re.findall(r"[A-Z]+", pre_suffix_name))
    is_business = bool(
        post_suffix_words & BUSINESS_KEYWORDS
        or pre_suffix_words & POLICY_ORGANIZATION_KEYWORDS
        or any(
            word.startswith(stem)
            for word in pre_suffix_words
            for stem in POLICY_ORGANIZATION_STEMS
        )
        or any(
            phrase in pre_suffix_name
            for phrase in POLICY_ORGANIZATION_PHRASES
        )
        or bool(re.search(r"(?:^|\s)P\.?A\.?$", pre_suffix_name))
    )
    return name, is_business


def _strip_stacked_property_metadata(value: str) -> str:
    """Remove every trailing legal wrapper, independent of list ordering."""
    name = value.strip()
    while name:
        for suffix in PROPERTY_METADATA_SUFFIXES:
            if name == suffix:
                name = ""
                break
            marker = " " + suffix
            if name.endswith(marker):
                name = name[: -len(marker)].rstrip()
                break
        else:
            return name
    return name


def _owner_segments(value: str, comma_mode: str = "auto") -> tuple[str, ...]:
    """Split explicit multi-person fields without guessing on whitespace."""
    if comma_mode == "auto":
        comma_segments = _comma_owner_segments(value)
        if comma_segments:
            return comma_segments
    normalized, _ = _normalize_name(value)
    if not normalized:
        return ()
    segments = tuple(
        segment.strip()
        for segment in OWNER_SEGMENT_RE.split(normalized)
        if segment.strip()
    )
    if segments:
        return segments
    return () if OWNER_SEGMENT_RE.search(normalized) else (normalized,)


def _has_sequence(words: list[str], sequence: tuple[str, ...]) -> bool:
    return any(tuple(words[index : index + len(sequence)]) == sequence
               for index in range(len(words) - len(sequence) + 1))


def _compound_positions(words: list[str]) -> set[int]:
    positions: set[int] = set()
    for sequence in COMPOUND_SEQUENCES:
        for index in range(len(words) - len(sequence) + 1):
            if tuple(words[index : index + len(sequence)]) == sequence:
                positions.update(range(index, index + len(sequence)))
    return positions


def _country_entry_strength(entry: dict | None) -> tuple[float, float, float]:
    if not entry:
        return 0.0, 0.0, 0.0
    weights = entry.get("combined_weights") or entry.get("weights") or {}
    counts = entry.get("counts") or {}
    lusophone_weight = sum(float(weights.get(country, 0.0)) for country in LUSOPHONE_CLASSES)
    lusophone_count = sum(float(counts.get(country, 0.0)) for country in LUSOPHONE_CLASSES)
    return lusophone_weight, float(weights.get("hispanic", 0.0)), lusophone_count


def _strong_surname(token: str, frequencies: dict, countries: dict) -> bool:
    if token in AMBIGUOUS_BARE_SURNAMES or token in COLLISION_MERGED_SURNAMES:
        return False
    if token in COUNTRY_CLUSTER_COLLISIONS:
        return False
    if token in MERGED_PREPS:
        return True
    if token in COUNTRY_CLUSTER_SURNAMES:
        # Curated country clusters are useful candidates, but some entries
        # (for example DELGADO/MORENO) are overwhelmingly Hispanic in the
        # underlying data.  A boost alone is not enough for High confidence.
        entry = countries.get("surnames", {}).get(token) or {}
        weights = entry.get("combined_weights") or entry.get("weights") or {}
        lusophone_weight = sum(
            float(weights.get(country, 0.0))
            for country in LUSOPHONE_CLASSES
        )
        if (
            lusophone_weight >= 0.45
            and float(weights.get("hispanic", 0.0)) < 0.35
        ):
            return True
    surname = frequencies.get("surnames", {}).get(token, {})
    if surname.get("brazilian", 0.0) >= 0.45 and surname.get("hispanic", 0.0) <= 0.60:
        return True
    luso_weight, hispanic_weight, luso_count = _country_entry_strength(
        countries.get("surnames", {}).get(token)
    )
    return luso_weight >= 0.85 and hispanic_weight <= 0.15 and luso_count >= 100


def _comma_owner_segments(value: str) -> tuple[str, ...]:
    """Split commas only when every side is a complete surname-led person.

    This preserves ordinary LAST, FIRST fields while isolating county formats
    such as ``SOUSA WILLIAM E, SOUSA JANET A``.
    """
    if "," not in value:
        return ()
    raw_parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(raw_parts) < 2:
        return ()
    frequencies, countries, _ = _policy_tables()
    segments: list[str] = []
    leading_surname = ""
    for part_index, part in enumerate(raw_parts):
        normalized, is_business = _normalize_name(part)
        if not normalized or is_business:
            return ()
        words = normalized.split()
        content_positions = [
            index
            for index, token in enumerate(words)
            if token not in POLICY_CONNECTORS
            and not (index > 0 and token in BR_SUFFIXES)
            and len(token) > 1
        ]
        if not content_positions:
            return ()
        first_token = words[content_positions[0]]
        if not _strong_surname(first_token, frequencies, countries):
            return ()
        if part_index == 0:
            if len(content_positions) < 2:
                return ()
            leading_surname = first_token
        elif len(content_positions) < 2 and first_token != leading_surname:
            return ()
        segments.append(normalized)
    return tuple(segments)


def _ambiguous_comma_parts(value: str) -> tuple[str, ...]:
    """Return complete comma sides that might instead be LAST, FIRST MIDDLE.

    County feeds use commas for both name order and, less consistently,
    co-owners.  When surname-led evidence is not strong on every side, no
    interpretation is reliable enough for High confidence.  The caller keeps
    the strongest independently scored side visible at Medium at most.
    """
    if "," not in value or _comma_owner_segments(value):
        return ()
    parts: list[str] = []
    for part_index, part in enumerate(item.strip() for item in value.split(",")):
        if not part:
            return ()
        normalized, is_business = _normalize_name(part)
        if not normalized or is_business:
            return ()
        words = normalized.split()
        content_count = sum(
            1
            for index, token in enumerate(words)
            if token not in POLICY_CONNECTORS
            and not (index > 0 and token in BR_SUFFIXES)
            and len(token) > 1
        )
        minimum_content = 2 if part_index == 0 else 1
        if content_count < minimum_content:
            return ()
        parts.append(normalized)
    return tuple(parts) if len(parts) >= 2 else ()


def _strong_given_name(token: str, frequencies: dict, countries: dict) -> bool:
    given = frequencies.get("first_names", {}).get(token)
    if given is not None:
        return (
            given.get("brazilian", 0.0) >= 0.65
            and given.get("hispanic", 0.0) <= 0.35
            and given.get("american", 0.0) <= 0.30
        )
    luso_weight, hispanic_weight, luso_count = _country_entry_strength(
        countries.get("given_names", {}).get(token)
    )
    return luso_weight >= 0.90 and hispanic_weight <= 0.10 and luso_count >= 50


def _corroborating_given_name(token: str, frequencies: dict) -> bool:
    """Frequency-backed given evidence safe enough to support Very High.

    Country-table-only fallback tokens are intentionally excluded: those
    tables do not include Haitian or South Asian negative classes and sparse
    entries such as DEMAS/CLAREL previously caused false corroboration.
    """
    given = frequencies.get("first_names", {}).get(token)
    return bool(
        given
        and given.get("brazilian", 0.0) >= 0.65
        and given.get("hispanic", 0.0) <= 0.35
        and given.get("american", 0.0) <= 0.30
    )


def _supporting_given_name(token: str, frequencies: dict) -> bool:
    """A Brazilian-weighted given signal suitable only for cautious High.

    This deliberately sits below the Very-High corroboration threshold, but
    still requires Brazil to lead both Hispanic and US-English frequencies by
    a material margin. It cannot support Very High on its own.
    """
    given = frequencies.get("first_names", {}).get(token)
    if not given:
        return False
    brazilian = float(given.get("brazilian", 0.0))
    return (
        brazilian >= 0.55
        and brazilian - float(given.get("hispanic", 0.0)) >= 0.20
        and brazilian - float(given.get("american", 0.0)) >= 0.20
    )


def _country_supported_shared_surname(token: str, countries: dict) -> bool:
    """Reconcile a Census-Hispanic token with substantial Brazil evidence.

    Shared Iberian surnames remain High-only: country evidence may clear a
    false Hispanic conflict, but it never supplies Very-High corroboration.
    """
    entry = countries.get("surnames", {}).get(token) or {}
    weights = entry.get("combined_weights") or entry.get("weights") or {}
    counts = entry.get("counts") or {}
    brazil_weight = float(weights.get("br", 0.0))
    return (
        float(counts.get("br", 0.0)) >= 100
        and brazil_weight >= 0.50
        and brazil_weight >= float(weights.get("hispanic", 0.0))
    )


def _reliable_brazilian_initial_surname(
    token: str,
    frequencies: dict,
    countries: dict,
) -> bool:
    if token in MERGED_PREPS:
        return True
    surname = frequencies.get("surnames", {}).get(token) or {}
    if (
        float(surname.get("brazilian", 0.0)) >= 0.45
        and float(surname.get("hispanic", 0.0)) <= 0.60
    ):
        return True
    entry = countries.get("surnames", {}).get(token) or {}
    weights = entry.get("combined_weights") or entry.get("weights") or {}
    counts = entry.get("counts") or {}
    return (
        float(counts.get("br", 0.0)) >= 100
        and float(weights.get("br", 0.0)) >= 0.50
    )


def _country_dominant_hispanic_given_token(
    token: str,
    frequencies: dict,
    countries: dict,
) -> bool:
    """Flag overwhelming Hispanic given-name evidence in a given-name role.

    The country table alone can mistake family names such as ROSSI for given
    names, so require agreement from the role-labelled frequency table and
    exclude tokens with strong competing family-name evidence.
    """
    if (
        token in COUNTRY_CLUSTER_SURNAMES
        or _country_supported_shared_surname(token, countries)
    ):
        return False
    if token in HIGH_CONFIDENCE_HISPANIC_GIVEN_OVERRIDES:
        return True
    given = frequencies.get("first_names", {}).get(token) or {}
    if (
        float(given.get("hispanic", 0.0)) < 0.75
        or float(given.get("brazilian", 0.0)) > 0.05
    ):
        return False
    surname = frequencies.get("surnames", {}).get(token) or {}
    likely_non_hispanic_family_name = (
        float(surname.get("american", 0.0)) >= 0.70
        and float(surname.get("hispanic", 0.0)) < 0.60
    )
    if likely_non_hispanic_family_name:
        return False
    entry = countries.get("given_names", {}).get(token) or {}
    weights = entry.get("combined_weights") or entry.get("weights") or {}
    counts = entry.get("counts") or {}
    lusophone_weight = sum(
        float(weights.get(country, 0.0)) for country in LUSOPHONE_CLASSES
    )
    return (
        float(weights.get("hispanic", 0.0)) >= 0.90
        and lusophone_weight <= 0.10
        and float(counts.get("hispanic", 0.0)) >= 1_000
    )


def _strong_country_lusophone_given_for_medium(
    token: str,
    countries: dict,
) -> bool:
    """Strong enough to undo only an API-surname veto, never reach High."""
    entry = countries.get("given_names", {}).get(token) or {}
    weights = entry.get("combined_weights") or entry.get("weights") or {}
    counts = entry.get("counts") or {}
    lusophone_weight = sum(
        float(weights.get(country, 0.0)) for country in LUSOPHONE_CLASSES
    )
    lusophone_count = sum(
        float(counts.get(country, 0.0)) for country in LUSOPHONE_CLASSES
    )
    return (
        lusophone_weight >= 0.70
        and float(weights.get("hispanic", 0.0)) <= 0.35
        and lusophone_count >= 50_000
    )


def _independent_cross_source_hispanic_given(
    token: str,
    frequencies: dict,
    countries: dict,
) -> bool:
    """A Hispanic given marker strong enough to resolve a mixed structure."""
    if token in HIGH_CONFIDENCE_HISPANIC_GIVEN_OVERRIDES:
        return True
    given = frequencies.get("first_names", {}).get(token)
    entry = countries.get("given_names", {}).get(token) or {}
    weights = entry.get("combined_weights") or entry.get("weights") or {}
    counts = entry.get("counts") or {}
    lusophone_weight = sum(
        float(weights.get(country, 0.0)) for country in LUSOPHONE_CLASSES
    )
    if given is not None:
        return bool(
            (
                float(given.get("hispanic", 0.0)) >= 0.55
                and float(given.get("brazilian", 0.0)) <= 0.35
                and float(weights.get("hispanic", 0.0)) >= 0.55
                and lusophone_weight <= 0.45
            )
            or (
                float(given.get("hispanic", 0.0)) >= 0.40
                and float(given.get("brazilian", 0.0)) <= 0.05
                and float(weights.get("hispanic", 0.0)) >= 0.85
                and lusophone_weight <= 0.15
            )
        )
    return (
        float(weights.get("hispanic", 0.0)) >= 0.95
        and float(counts.get("hispanic", 0.0)) >= 50
    )


def _definitive_hispanic_token(token: str, frequencies: dict) -> bool:
    if token in POLICY_SHARED_IBERIAN_SURNAMES or token in MERGED_PREPS:
        return False
    surname = frequencies.get("surnames", {}).get(token) or {}
    return (
        surname.get("hispanic", 0.0) >= 0.72
        and surname.get("brazilian", 0.0) < 0.35
    )


def _country_dominant_hispanic_surname(token: str, countries: dict) -> bool:
    if _country_supported_shared_surname(token, countries):
        return False
    entry = countries.get("surnames", {}).get(token) or {}
    weights = entry.get("combined_weights") or entry.get("weights") or {}
    counts = entry.get("counts") or {}
    return (
        float(weights.get("hispanic", 0.0)) >= 0.90
        and float(counts.get("hispanic", 0.0)) >= 100
    )


def _compound_family_conflict(
    token: str,
    frequencies: dict,
    conflicts: dict[str, dict[str, float | int]],
) -> bool:
    """Reject unsafe family tokens from compound-given-only rescues."""
    surname = frequencies.get("surnames", {}).get(token) or {}
    if (
        surname.get("hispanic", 0.0) >= 0.60
        and surname.get("brazilian", 0.0) < 0.35
    ):
        return True
    signal = conflicts.get(token) or {}
    return bool(
        int(signal.get("count", 0)) >= 100
        and (
            float(signal.get("pctapi", 0.0)) >= 70.0
            or float(signal.get("pctblack", 0.0)) >= 85.0
        )
    )


def precision_evidence(value: str) -> PrecisionEvidence:
    """Return the conservative evidence tier for one owner-name field."""
    frequencies, countries, conflicts = _policy_tables()
    has_explicit_last_first_separator = "," in value
    normalized, is_business = _normalize_name(value)
    if not normalized or is_business:
        return PrecisionEvidence(0, None, is_business, ())

    words = normalized.split()
    substantive_words = _strip_stacked_property_metadata(normalized).split()
    substantive_content_count = sum(
        1
        for index, token in enumerate(substantive_words)
        if token not in POLICY_CONNECTORS
        and not (index > 0 and token in BR_SUFFIXES)
        and len(token) > 1
    )
    strong_portuguese_suffix_present = any(
        index > 0 and token in STRONG_PORTUGUESE_SUFFIXES
        for index, token in enumerate(substantive_words)
    )
    content_positions = [
        index
        for index, token in enumerate(words)
        if token not in POLICY_CONNECTORS
        and not (index > 0 and token in BR_SUFFIXES)
        and len(token) > 1
    ]
    if not content_positions:
        return PrecisionEvidence(0, None, False, ())

    compound_positions = _compound_positions(words)
    surname_positions: set[int] = set()

    # A validated connector phrase is one surname unit. It may establish a
    # surname role, but its connector never counts as independent evidence.
    leading_compound = False
    validated_compound_present = False
    for sequence in VALID_COMPOUND_SEQUENCES:
        for index in range(len(words) - len(sequence) + 1):
            if tuple(words[index : index + len(sequence)]) == sequence:
                validated_compound_present = True
                surname_positions.add(index + len(sequence) - 1)
                if index == 0:
                    leading_compound = True

    # Property appraisers predominantly emit LAST FIRST. Accept an initial
    # coherent surname run, including validated compound units.
    for cursor in content_positions:
        if cursor in compound_positions:
            break
        if cursor in surname_positions:
            continue
        if not _strong_surname(words[cursor], frequencies, countries):
            break
        if (
            _strong_given_name(words[cursor], frequencies, countries)
            and words[cursor] not in frequencies.get("surnames", {})
        ):
            break
        surname_positions.add(cursor)

    # Also support natural FIRST ... LAST order, but only when an earlier,
    # separate token has frequency-backed given-name evidence. Country-table
    # fallback is insufficient because its negative classes are incomplete.
    last_position = content_positions[-1]
    earlier_given = any(
        position != last_position
        and _corroborating_given_name(words[position], frequencies)
        and not _strong_surname(words[position], frequencies, countries)
        for position in content_positions
    )
    if (
        earlier_given
        and last_position not in compound_positions
        and _strong_surname(words[last_position], frequencies, countries)
    ):
        surname_positions.add(last_position)

    connector_anchored = False
    for index, token in enumerate(words):
        if token not in POLICY_CONNECTORS:
            continue
        next_position = next((position for position in content_positions if position > index), None)
        if next_position is not None and _strong_surname(words[next_position], frequencies, countries):
            surname_positions.add(next_position)
            connector_anchored = True

    merged_surname_present = False
    for position in content_positions:
        if (
            words[position] in MERGED_PREPS
            and words[position] not in COLLISION_MERGED_SURNAMES
        ):
            surname_positions.add(position)
            merged_surname_present = True

    has_ambiguous_de_silva = any(
        _has_sequence(words, sequence)
        for sequence in AMBIGUOUS_COMPOUND_SURNAMES
    )
    has_merged_collision = any(
        token in COLLISION_MERGED_SURNAMES
        for token in words
    )
    if has_ambiguous_de_silva or has_merged_collision:
        for position in content_positions:
            token = words[position]
            if (
                token in COLLISION_MERGED_SURNAMES
                or (has_ambiguous_de_silva and token == "SILVA")
            ):
                continue
            if _strong_surname(token, frequencies, countries):
                surname_positions.add(position)

    role_surname_positions = set(surname_positions)
    scan_candidates = {
        position
        for position in content_positions
        if position not in compound_positions
        and words[position] not in AMBIGUOUS_BARE_SURNAMES
        and words[position] not in COLLISION_MERGED_SURNAMES
        and _strong_surname(words[position], frequencies, countries)
    }
    tentative_surnames = surname_positions | scan_candidates
    tentative_unique_surnames = {
        words[position] for position in tentative_surnames
    }
    tentative_given_support = any(
        position not in tentative_surnames
        and _corroborating_given_name(words[position], frequencies)
        for position in content_positions
    )
    compound_given = any(_has_sequence(words, sequence) for sequence in COMPOUND_SEQUENCES)
    cautious_natural_order = bool(
        not has_explicit_last_first_separator
        and content_positions[0] not in scan_candidates
        and _supporting_given_name(words[content_positions[0]], frequencies)
        and any(position > content_positions[0] for position in scan_candidates)
    )
    coherent_scan_evidence = bool(
        len(tentative_unique_surnames) >= 2
        or (tentative_surnames and (tentative_given_support or compound_given))
    )
    natural_order_rescue = cautious_natural_order and not coherent_scan_evidence
    if coherent_scan_evidence or cautious_natural_order:
        surname_positions.update(scan_candidates)
    scan_changed = surname_positions != role_surname_positions

    given_support = any(
        position not in surname_positions
        and _corroborating_given_name(words[position], frequencies)
        for position in content_positions
    )
    brazilian_compound_given = any(
        _has_sequence(words, sequence) for sequence in BR_COMPOUND_SEQUENCES
    )
    compound_family_positions = [
        position
        for position in content_positions
        if position not in compound_positions
    ]
    compound_family_conflict = any(
        _compound_family_conflict(words[position], frequencies, conflicts)
        for position in compound_family_positions
    )

    unique_surname_count = len({words[position] for position in surname_positions})
    if unique_surname_count >= 2 or (
        surname_positions
        and (given_support or compound_given)
    ):
        tier = 2
    elif surname_positions:
        tier = 1
    else:
        tier = 0

    # The raw>=70 guard is needed only when a lone surname was discovered in
    # an uncertain later role. Coherent evidence (two surnames, or one surname
    # plus a frequency-backed given name) is independently meaningful.
    guarded_rescue = scan_changed and tier == 1
    high_only = (
        (
            scan_changed
            and (tier == 1 or (tier == 2 and unique_surname_count == 1))
        )
        or (
            tier == 1
            and (
                validated_compound_present
                or connector_anchored
                or (merged_surname_present and len(content_positions) >= 2)
                or words[0] in STRONG_PORTUGUESE_SUFFIXES
                or any(
                    index > 0 and token in STRONG_PORTUGUESE_SUFFIXES
                    for index, token in enumerate(words)
                )
            )
        )
    )
    first_content_position = content_positions[0]
    supporting_given_rescue = bool(
        tier == 1
        and first_content_position in surname_positions
        and _reliable_brazilian_initial_surname(
            words[first_content_position], frequencies, countries
        )
        and any(
            position > first_content_position
            and position not in surname_positions
            and _supporting_given_name(words[position], frequencies)
            and float(
                (
                    countries.get("given_names", {}).get(words[position], {})
                    .get("counts", {})
                ).get("br", 0.0)
            )
            >= 10_000
            for position in content_positions
        )
    )
    if supporting_given_rescue:
        guarded_rescue = True
        high_only = True

    # Brazil has large Italian/German/Japanese-descendant populations. A
    # specifically Brazilian compound given name plus a separate, neutral
    # family token is enough for a cautious High lead, but never Very High.
    if (
        tier == 0
        and brazilian_compound_given
        and compound_family_positions
        and not compound_family_conflict
    ):
        tier = 1
        guarded_rescue = True
        high_only = True

    # FILHO/NETO/SOBRINHO are useful Portuguese structure only when a real
    # multi-token name remains after removing the suffix. JUNIOR is excluded
    # because it is broadly cross-cultural.
    if (
        tier == 0
        and any(
            index > 0 and token in STRONG_PORTUGUESE_SUFFIXES
            for index, token in enumerate(words)
        )
        and (
            len(content_positions) >= 2
            or (
                len(content_positions) == 1
                and _corroborating_given_name(
                    words[content_positions[0]], frequencies
                )
            )
        )
    ):
        tier = 1
        guarded_rescue = True
        high_only = True

    # A leading apparent Portuguese particle must still be checked as a
    # property-record surname. Only a validated compound plus separate
    # given-name support disambiguates it as contextual syntax.
    census_conflict: str | None = None
    hispanic_conjunction = any(
        token == "Y"
        and 0 < index < len(words) - 1
        and (index - 1) in scan_candidates
        and any(position > index for position in scan_candidates)
        for index, token in enumerate(words)
    )
    if (
        any(
            _has_sequence(words, tuple(phrase.split()))
            for phrase in HISPANIC_STRUCTURE_PHRASES
        )
        or "DEL" in words
        or any(token in MERGED_HISPANIC_STRUCTURES for token in words)
        or hispanic_conjunction
    ):
        census_conflict = "hispanic"
    if (
        census_conflict is None
        and tier >= 1
        and any(
            position not in surname_positions
            and position not in compound_positions
            and not (
                position == len(words) - 1
                and len(_strip_accents(value).strip()) == 30
                and words[position] in TRUNCATED_PROPERTY_TOKEN_PREFIXES
                and _strip_accents(value).upper().strip().endswith(words[position])
            )
            and _country_dominant_hispanic_given_token(
                words[position], frequencies, countries
            )
            for position in content_positions
        )
    ):
        census_conflict = "hispanic_given"
    validated_leading_particle = leading_compound and (given_support or compound_given)
    first_signal = conflicts.get(words[0])
    if census_conflict is None and first_signal and not validated_leading_particle:
        signal_count = int(first_signal.get("count", 0))
        if signal_count >= 100 and float(first_signal.get("pctapi", 0.0)) >= 70.0:
            census_conflict = "api"
        elif signal_count >= 100 and float(first_signal.get("pctblack", 0.0)) >= 85.0:
            census_conflict = "black"
    api_medium_rescue = bool(
        census_conflict == "api"
        and (
            _strong_country_lusophone_given_for_medium(words[0], countries)
            or any(
                position > 0
                and _strong_surname(words[position], frequencies, countries)
                for position in content_positions
            )
        )
    )
    # A first token can be a given name that also appears in the Census API
    # surname aggregate (BIBI/TIAO). A later supported surname may clear only
    # the API conflict, never a Black conflict, and the result remains High.
    contextual_api_override = bool(
        census_conflict == "api"
        and words[0] not in POLICY_CONNECTORS
        and _corroborating_given_name(words[0], frequencies)
        and any(position > 0 for position in surname_positions)
    )
    if contextual_api_override:
        census_conflict = None
        guarded_rescue = True
        high_only = True

    if (
        census_conflict is None
        and (has_ambiguous_de_silva or has_merged_collision)
        and not any(
            words[position] != "SILVA"
            and words[position] not in COLLISION_MERGED_SURNAMES
            for position in surname_positions
        )
    ):
        census_conflict = "ambiguous"
    if (
        census_conflict is None
        and tier >= 1
    ):
        # Distinct Spanish surnames cannot borrow a Portuguese-looking token
        # elsewhere in the same person segment (e.g. ARIAS MOREIRA LUANA).
        country_supported_shared_positions = {
            position
            for position in content_positions
            if position not in surname_positions
            and position not in compound_positions
            and words[position] not in frequencies.get("first_names", {})
            and _definitive_hispanic_token(words[position], frequencies)
            and _country_supported_shared_surname(words[position], countries)
        }
        definitive_hispanic_positions = {
            position
            for position in content_positions
            if position not in surname_positions
            and position not in compound_positions
            and words[position] not in frequencies.get("first_names", {})
            and _definitive_hispanic_token(words[position], frequencies)
            and not _country_supported_shared_surname(words[position], countries)
        }
        has_country_supported_shared_surname = bool(
            country_supported_shared_positions
        )
        has_definitive_hispanic = bool(definitive_hispanic_positions)
        has_independent_hispanic_given = any(
            position not in surname_positions
            and position not in compound_positions
            and position not in definitive_hispanic_positions
            and _independent_cross_source_hispanic_given(
                words[position], frequencies, countries
            )
            for position in content_positions
        )
        if has_country_supported_shared_surname and not validated_compound_present:
            high_only = True
        # A validated Portuguese compound can coexist with a shared family
        # token, but not with both a definitive Spanish surname and an
        # independent cross-source Hispanic given marker.
        validated_lusophone_context = (
            validated_compound_present and not has_independent_hispanic_given
        )
        if has_definitive_hispanic and not validated_lusophone_context:
            census_conflict = "hispanic"

    if census_conflict is None and tier >= 1:
        for position in content_positions:
            token = words[position]
            if token == "LE" and position == len(words) - 1:
                # A bare terminal LE is life-estate metadata in 99.5% of the
                # matching stored rows, but can be a surname elsewhere. It
                # clears only this veto and caps the result at High.
                high_only = True
                continue
            if (
                position in surname_positions
                or token in frequencies.get("first_names", {})
                or _strong_surname(token, frequencies, countries)
            ):
                continue
            secondary_signal = conflicts.get(token)
            if (
                secondary_signal
                and int(secondary_signal.get("count", 0)) >= 100
                and max(
                    float(secondary_signal.get("pctapi", 0.0)),
                    float(secondary_signal.get("pctblack", 0.0)),
                ) >= 90.0
            ):
                census_conflict = "secondary"
                break

    surname_tokens = tuple(
        dict.fromkeys(words[position] for position in sorted(surname_positions))
    )
    country_hispanic_surname_conflict = bool(
        tier >= 1
        and any(
            position not in surname_positions
            and position not in compound_positions
            and words[position] not in frequencies.get("first_names", {})
            and _country_dominant_hispanic_surname(
                words[position], countries
            )
            for position in content_positions
        )
    )
    floor_eligible = tier == 2 and (
        unique_surname_count >= 2
        or given_support
        or compound_given
    )
    return PrecisionEvidence(
        tier,
        census_conflict,
        False,
        surname_tokens,
        guarded_rescue,
        high_only,
        floor_eligible,
        natural_order_rescue,
        supporting_given_rescue,
        api_medium_rescue,
        country_hispanic_surname_conflict,
        substantive_content_count,
        strong_portuguese_suffix_present,
    )


def _evidence_cap(
    evidence: PrecisionEvidence,
    raw_score: int = 100,
    probabilities: dict[str, float] | None = None,
) -> tuple[int, str]:
    if evidence.organization:
        return 0, "organization_name_excluded"
    if (
        evidence.census_conflict == "api"
        and evidence.api_medium_rescue
        and float((probabilities or {}).get("brazilian", 0.0)) >= 0.90
    ):
        return 49, "api_conflict_medium_only"
    if evidence.census_conflict == "api":
        return 14, "asian_pacific_surname_conflict_cap"
    if evidence.census_conflict == "black":
        return 49, "black_surname_conflict_cap"
    if evidence.census_conflict == "hispanic":
        return 49, "definitive_hispanic_surname_conflict_cap"
    if evidence.census_conflict == "hispanic_given":
        return 49, "country_dominant_hispanic_given_conflict_cap"
    if evidence.census_conflict == "ambiguous":
        return 49, "ambiguous_de_silva_collision_cap"
    if evidence.census_conflict == "secondary":
        return 49, "secondary_cross_cultural_token_conflict_cap"
    if (
        evidence.country_hispanic_surname_conflict
        and float((probabilities or {}).get("hispanic", 0.0))
        >= float((probabilities or {}).get("brazilian", 0.0))
    ):
        return 49, "country_dominant_hispanic_surname_conflict_cap"
    if (
        raw_score >= 50
        and len(set(evidence.surname_tokens)) <= 1
        and float((probabilities or {}).get("hispanic", 0.0))
        > float((probabilities or {}).get("brazilian", 0.0))
    ):
        return 49, "high_requires_brazilian_over_hispanic_model"
    if evidence.guarded_rescue and raw_score < 70:
        return 49, "role_rescue_requires_raw_very_high"
    if evidence.natural_order_rescue:
        probabilities = probabilities or {}
        if (
            float(probabilities.get("brazilian", 0.0)) < 0.90
            or float(probabilities.get("hispanic", 1.0)) > 0.40
            or float(probabilities.get("american", 1.0)) > 0.40
        ):
            return 49, "natural_order_rescue_requires_strong_model_support"
    if evidence.supporting_given_rescue:
        probabilities = probabilities or {}
        if (
            float(probabilities.get("brazilian", 0.0)) < 0.90
            or float(probabilities.get("hispanic", 1.0)) > 0.40
            or float(probabilities.get("american", 1.0)) > 0.40
        ):
            return 49, "supporting_given_rescue_requires_strong_model_support"
    if evidence.tier == 0:
        return 49, "high_requires_lusophone_surname_evidence"
    if evidence.tier == 1 and not evidence.high_only:
        return 49, "high_requires_corroborated_or_structural_evidence"
    if (
        evidence.tier == 1
        and evidence.substantive_content_count <= 1
        and not evidence.strong_portuguese_suffix_present
    ):
        return 49, "high_requires_person_level_evidence"
    if (
        raw_score >= 70
        and evidence.country_hispanic_surname_conflict
        and float((probabilities or {}).get("hispanic", 0.0)) >= 0.50
    ):
        return 69, "very_high_hispanic_surname_model_conflict"
    if evidence.tier == 1 or evidence.high_only:
        return 69, "very_high_requires_corroborated_lusophone_evidence"
    if (
        raw_score >= 70
        and probabilities
        and float(probabilities.get("brazilian", 0.0))
        < max(
            float(probabilities.get("hispanic", 0.0)),
            float(probabilities.get("american", 0.0)),
        )
    ):
        return 69, "very_high_requires_brazilian_model_lead"
    return 100, ""


def _confidence_label(score: int) -> str:
    if score >= 70:
        return "Very High"
    if score >= 50:
        return "High"
    if score >= 30:
        return "Medium"
    if score >= 15:
        return "Low"
    return "Very Low"


def _apply_precision_policy(result, name: str, comma_mode: str = "auto"):
    evidence = precision_evidence(name)
    segments = _owner_segments(name, comma_mode)
    normalized_name, _ = _normalize_name(name)
    has_explicit_owner_delimiter = bool(OWNER_SEGMENT_RE.search(normalized_name))
    if has_explicit_owner_delimiter and not segments:
        return replace(
            result,
            score=0,
            confidence=_confidence_label(0),
            reasons=["empty_owner_segment_excluded"],
        )
    if len(segments) > 1 or (segments and has_explicit_owner_delimiter):
        segment_results = [
            _apply_precision_policy(
                scorer.classify_name(segment), segment, comma_mode=comma_mode
            )
            for segment in segments
        ]
        strongest = max(segment_results, key=lambda item: item.score)
        if evidence.organization and strongest.score < 50:
            return replace(
                result,
                score=0,
                confidence=_confidence_label(0),
                reasons=["organization_name_excluded"],
            )
        reasons = list(strongest.reasons)
        reason = "multi_person_field_requires_individual_evidence"
        if reason not in reasons:
            reasons.append(reason)
        return replace(
            strongest,
            score=strongest.score,
            confidence=_confidence_label(strongest.score),
            reasons=reasons,
        )

    if evidence.organization:
        return replace(
            result,
            score=0,
            confidence=_confidence_label(0),
            reasons=["organization_name_excluded"],
        )

    ambiguous_comma_parts = (
        _ambiguous_comma_parts(name) if comma_mode == "auto" else ()
    )
    if ambiguous_comma_parts:
        comma_results = [
            _apply_precision_policy(
                scorer.classify_name(part), part, comma_mode="auto"
            )
            for part in ambiguous_comma_parts
        ]
        strongest = max(comma_results, key=lambda item: item.score)
        score = min(strongest.score, 49)
        reasons = list(strongest.reasons)
        reason = "ambiguous_comma_requires_manual_review_cap"
        if reason not in reasons:
            reasons.append(reason)
        return replace(
            strongest,
            score=score,
            confidence=_confidence_label(score),
            reasons=reasons,
        )

    cap, reason = _evidence_cap(evidence, result.score, result.probabilities)
    if result.score > cap:
        reasons = list(result.reasons)
        if reason not in reasons:
            reasons.append(reason)
        return replace(
            result,
            score=cap,
            confidence=_confidence_label(cap),
            reasons=reasons,
        )

    if (
        result.score < 50
        and evidence.floor_eligible
        and not evidence.guarded_rescue
        and evidence.census_conflict is None
        and not evidence.organization
        and not any(reason.endswith("_cap") for reason in result.reasons)
        and float(result.probabilities.get("brazilian", 0.0))
        >= float(result.probabilities.get("hispanic", 1.0)) + 0.05
        and float(result.probabilities.get("brazilian", 0.0))
        >= float(result.probabilities.get("american", 1.0)) + 0.05
    ):
        reasons = list(result.reasons)
        reason = "corroborated_lusophone_evidence_floor"
        if reason not in reasons:
            reasons.append(reason)
        return replace(
            result,
            score=50,
            confidence=_confidence_label(50),
            reasons=reasons,
        )

    return result


def _validate_comma_mode(comma_mode: str) -> None:
    if comma_mode not in {"auto", "last_first"}:
        raise ValueError(f"unsupported comma mode: {comma_mode}")


def _annotate_comma_mode(result, name: str, comma_mode: str):
    if comma_mode != "last_first" or "," not in name:
        return result
    reasons = list(result.reasons)
    reason = "source_confirmed_last_first_comma_format"
    if reason not in reasons:
        reasons.append(reason)
    return replace(result, reasons=reasons)


def classify_name(name: str, comma_mode: str = "auto"):
    _validate_comma_mode(comma_mode)
    result = _apply_precision_policy(
        scorer.classify_name(name), name, comma_mode=comma_mode
    )
    return _annotate_comma_mode(result, name, comma_mode)


def classify_record(
    name1: str,
    name2: str = "",
    comma_mode: str = "auto",
):
    _validate_comma_mode(comma_mode)
    first = classify_name(name1, comma_mode=comma_mode)
    if not name2:
        return first

    second = classify_name(name2, comma_mode=comma_mode)
    strongest = first if first.score >= second.score else second
    reasons = list(strongest.reasons)
    reason = "record_score_uses_strongest_independently_qualified_owner"
    if reason not in reasons:
        reasons.append(reason)
    return replace(strongest, reasons=reasons)
