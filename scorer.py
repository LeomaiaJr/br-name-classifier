"""BR Name Classifier scoring API.

Uses census frequency tables + character n-gram TF-IDF + meta-classifier
to produce a 0-100 Brazilian probability score with match reasons.

Usage:
    from scorer import classify_name, classify_record

    result = classify_name("FERREIRA GUSTAVO DA SILVA")
    print(result.score, result.confidence, result.reasons)
"""

import json
import math
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    PT_STRONG_PREPS, PT_WEAK_PREPS, ALL_PT_PREPS, ES_PATTERNS,
    BR_SUFFIXES, MERGED_PREPS, COMPOUND_FIRST_NAMES, BUSINESS_KEYWORDS,
    SUFFIX_STRIP, CONFIDENCE_THRESHOLDS, FEATURE_NAMES,
    CURATED_BR_GIVEN_NAMES, CURATED_PT_GIVEN_NAMES, CURATED_CV_SURNAMES,
    CURATED_AO_TOKENS, CURATED_MZ_SURNAMES,
)

_BASE = Path(__file__).parent
_OUTPUT_DIR = _BASE / "output"
_MODELS_DIR = _BASE / "models"

# Lazy-loaded model cache
_freq_tables = None
_ngram_pipeline = None
_meta_coef = None
_meta_intercept = None
_us_census_pcthisp = None
_country_tables = None

_SPANISH_PREP_SEQS = tuple(tuple(p.split()) for p in ES_PATTERNS)
_COUNTRY_CLASSES = ("br", "pt", "cv", "ao", "mz", "palop_other")
_COUNTRY_EVIDENCE_CLASSES = _COUNTRY_CLASSES + ("hispanic",)
_DEFAULT_COUNTRY_PRIOR = {
    "br": 0.45,
    "pt": 0.25,
    "cv": 0.10,
    "ao": 0.08,
    "mz": 0.07,
    "palop_other": 0.05,
}
_SHARED_LUSOPHONE_TOKENS = frozenset({
    "SILVA", "SANTOS", "PEREIRA", "FERREIRA", "GOMES", "LOPES",
    "RODRIGUES", "COSTA", "MARTINS", "FERNANDES", "JOSE", "MARIA",
    "JOAO", "ANTONIO", "FRANCISCO", "ANA", "PEDRO", "MANUEL", "PAULO",
    "MIGUEL", "OLIVEIRA", "SOUSA", "SOUZA", "RIBEIRO",
})
_HAITIAN_FRENCH_TOKENS = frozenset({
    "JEAN", "MARIE", "PIERRE", "FRANCOIS", "ETIENNE", "JOSEPH", "LOUIS",
    "JACQUES", "SAINT", "ST", "FLEUR", "PHANOR", "BEAUBRUN", "CADET",
    "TOUSSAINT", "DESSALIN", "DESSALINES", "DESNOYERS", "DORIVAL",
    "CAJUSTE", "CLERIE", "LAFORTUNE", "BOUCICAUT", "ANTONINE", "ANTOINE",
})
_PORTUGUESE_SHARED_HISPANIC_SURNAMES = frozenset({
    "ALVES", "ANDRADE", "BATISTA", "CHAVES", "FERNANDES", "FONSECA", "GOMES",
    "GONCALVES", "LOPES", "MARQUES", "MENDES", "NUNES", "PIRES",
    "RODRIGUES", "SOARES",
})


@dataclass
class ClassificationResult:
    score: int = 0
    confidence: str = "Very Low"
    reasons: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    country_probs: dict[str, float] = field(default_factory=dict)
    country_entropy: float = 0.0


def _load_models():
    global _freq_tables, _ngram_pipeline, _meta_coef, _meta_intercept, _us_census_pcthisp, _country_tables

    if _freq_tables is not None:
        return

    with open(_OUTPUT_DIR / "frequency_tables.json") as f:
        _freq_tables = json.load(f)

    country_path = _OUTPUT_DIR / "country_tables.json"
    if country_path.exists():
        with open(country_path) as f:
            _country_tables = json.load(f)
    else:
        _country_tables = {}

    with open(_MODELS_DIR / "ngram_pipeline.pkl", "rb") as f:
        _ngram_pipeline = pickle.load(f)

    with open(_OUTPUT_DIR / "meta_model.json") as f:
        meta = json.load(f)
    _meta_coef = np.array(meta["coef"])
    _meta_intercept = meta["intercept"]

    pcthisp_path = _BASE / "data" / "processed" / "all_surnames.csv"
    _us_census_pcthisp = {}
    if pcthisp_path.exists():
        df = pd.read_csv(pcthisp_path)
        for _, row in df[df["country"] == "american"].iterrows():
            try:
                pct = float(row["pcthispanic"])
                if pct > 0:
                    _us_census_pcthisp[str(row["name"])] = pct
            except (ValueError, TypeError):
                pass


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_name(name: str) -> tuple[str, bool]:
    """Returns (cleaned_name, is_business_entity)."""
    if not name:
        return "", False
    name = _strip_accents(name).upper().strip()
    name = name.replace(",", " ")
    name = name.replace("-", " ")
    name = re.sub(r"\bL\s*\.?\s*L\s*\.?\s*C\b\.?", "LLC", name)
    name = re.sub(r"^REM:\s*", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name.startswith("THE "):
        name = name[4:].strip()
    for suffix in SUFFIX_STRIP:
        if name.endswith(" " + suffix):
            name = name[: -(len(suffix) + 1)]
    if set(name.split()) & BUSINESS_KEYWORDS:
        return name, True
    return re.sub(r"\s+", " ", name).strip(), False


def _has_sequence(words: list[str], seq: tuple[str, ...]) -> bool:
    if len(seq) > len(words):
        return False
    return any(tuple(words[i:i + len(seq)]) == seq for i in range(len(words) - len(seq) + 1))


def _has_spanish_preposition(words: list[str]) -> bool:
    return any(_has_sequence(words, seq) for seq in _SPANISH_PREP_SEQS)


def _has_compound_first_name(words: list[str]) -> bool:
    compounds = {tuple(cn.split()) for cn in COMPOUND_FIRST_NAMES}
    return any(_has_sequence(words, seq) for seq in compounds)


def _is_generational_suffix(words: list[str], index: int) -> bool:
    return words[index] in BR_SUFFIXES and index > 0


def _role_slots(words: list[str]) -> tuple[list[str], list[str]]:
    """Split property-record names into likely surname and given-name slots."""
    if not words:
        return [], []

    content_positions = [i for i, _ in enumerate(words) if not _is_generational_suffix(words, i)]
    if not content_positions:
        return [], []

    surname_positions: set[int] = set()
    prep_positions = {i for i, w in enumerate(words) if w in ALL_PT_PREPS}

    first_pos = content_positions[0]
    if words[first_pos] in ALL_PT_PREPS:
        for pos in content_positions[1:]:
            if words[pos] not in ALL_PT_PREPS:
                surname_positions.add(pos)
                break
    else:
        surname_positions.add(first_pos)

    for pos in prep_positions:
        prev_pos = next((i for i in range(pos - 1, -1, -1) if words[i] not in ALL_PT_PREPS and not _is_generational_suffix(words, i)), None)
        next_pos = next((i for i in range(pos + 1, len(words)) if words[i] not in ALL_PT_PREPS and not _is_generational_suffix(words, i)), None)
        if prev_pos is not None:
            surname_positions.add(prev_pos)
        if next_pos is not None:
            surname_positions.add(next_pos)

    surname_tokens = [
        words[i] for i in sorted(surname_positions)
        if words[i] not in ALL_PT_PREPS and not _is_generational_suffix(words, i)
    ]
    firstname_tokens = [
        words[i] for i in content_positions
        if i not in surname_positions and i not in prep_positions and not _is_generational_suffix(words, i)
    ]
    return surname_tokens, firstname_tokens


def _country_prior() -> dict[str, float]:
    prior = (_country_tables or {}).get("unattributable_prior") or _DEFAULT_COUNTRY_PRIOR
    return _normalize_country_probs({c: float(prior.get(c, 0.0)) for c in _COUNTRY_CLASSES})


def _normalize_country_probs(probs: dict[str, float]) -> dict[str, float]:
    clean = {c: max(0.0, float(probs.get(c, 0.0))) for c in _COUNTRY_CLASSES}
    total = sum(clean.values())
    if total <= 0:
        clean = dict(_DEFAULT_COUNTRY_PRIOR)
        total = sum(clean.values())
    return {c: clean[c] / total for c in _COUNTRY_CLASSES}


def _normalize_country_evidence_weights(probs: dict[str, float]) -> dict[str, float]:
    clean = {c: max(0.0, float(probs.get(c, 0.0))) for c in _COUNTRY_EVIDENCE_CLASSES}
    total = sum(clean.values())
    if total <= 0:
        return {c: 0.0 for c in _COUNTRY_EVIDENCE_CLASSES}
    return {c: clean[c] / total for c in _COUNTRY_EVIDENCE_CLASSES}


def _country_entropy(probs: dict[str, float]) -> float:
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return entropy / math.log(len(_COUNTRY_CLASSES))


def _ranked_country_weights(weights: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(
        ((c, float(weights.get(c, 0.0))) for c in _COUNTRY_CLASSES),
        key=lambda item: (-item[1], item[0]),
    )


def _is_country_specific(weights: dict[str, float], token: str, role: str) -> str | None:
    ranked = _ranked_country_weights(weights)
    ranked = sorted(
        ((c, float(weights.get(c, 0.0))) for c in _COUNTRY_EVIDENCE_CLASSES),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] <= 0:
        return None
    top_country, top_weight = ranked[0]
    second_weight = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_country == "hispanic":
        return None
    if float(weights.get("hispanic", 0.0)) >= 0.35 and float(weights.get("hispanic", 0.0)) >= 0.75 * top_weight:
        return None
    if top_weight <= 2 * max(second_weight, 1e-9):
        return None
    if token in _SHARED_LUSOPHONE_TOKENS:
        return None
    if top_country == "ao":
        return "ao" if token in CURATED_AO_TOKENS else None
    if top_country == "pt":
        # Portugal is a soft signal: allow PT-heavy given names to affect
        # attribution, but the country-probability cap still prevents >0.70
        # claims from whitelist evidence alone.
        return "pt" if role == "given" and top_weight >= 0.60 else None
    if top_country == "br":
        return "br" if role == "given" and token in CURATED_BR_GIVEN_NAMES else None
    if top_country == "cv" and role != "surname":
        return None
    if top_country == "mz" and role != "surname":
        return None
    return top_country


def _role_country_evidence(words: list[str]) -> tuple[dict[str, float], list[dict[str, object]]]:
    if not _country_tables:
        return _country_prior(), []

    surname_slots, firstname_slots = _role_slots(words)
    evidence: list[dict[str, object]] = []
    totals = {c: 0.0 for c in _COUNTRY_CLASSES}

    def add_token(token: str, role: str, role_weight: float) -> None:
        section = "surnames" if role == "surname" else "given_names"
        entry = (_country_tables.get(section) or {}).get(token)
        if not entry:
            return
        weights = entry.get("combined_weights") or entry.get("weights") or {}
        normalized = _normalize_country_evidence_weights(weights)
        specific = _is_country_specific(normalized, token, role)
        strength = max(normalized.values())
        for country, value in normalized.items():
            if country in _COUNTRY_CLASSES:
                totals[country] += role_weight * value
        evidence.append(
            {
                "token": token,
                "role": role,
                "weights": normalized,
                "specific": specific,
                "strength": strength,
            }
        )

    for token in surname_slots:
        add_token(token, "surname", 1.25)
        # Property records frequently arrive LAST FIRST, but single-token
        # ambiguity means we also let distinctive given names contribute.
        add_token(token, "given", 0.55)
    for token in firstname_slots:
        add_token(token, "given", 1.0)
        add_token(token, "surname", 0.55)

    # Phrase-level curated clusters that cannot be represented as one token.
    name = " ".join(words)
    for phrase in CURATED_CV_SURNAMES:
        if " " in phrase and phrase in name:
            totals["cv"] += 1.2
            evidence.append(
                {
                    "token": phrase,
                    "role": "surname",
                    "weights": _normalize_country_probs({"cv": 1.0}),
                    "specific": "cv",
                    "strength": 1.0,
                }
            )

    return totals, evidence


def _country_attribution_for_name(name: str) -> tuple[dict[str, float], float, list[str]]:
    words = name.split()
    prior = _country_prior()
    totals, evidence = _role_country_evidence(words)
    specific_countries = {item["specific"] for item in evidence if item.get("specific")}
    specific_countries.discard(None)

    if not evidence or not specific_countries:
        probs = prior
        entropy_value = _country_entropy(probs)
        return probs, entropy_value, ["country_attribution:shared_prior_high_entropy"]

    # Smooth toward the accepted Florida/Lusophone prior; token evidence then
    # moves only names with specific country evidence away from that mixture.
    smoothed = {c: prior[c] * 1.5 + totals.get(c, 0.0) for c in _COUNTRY_CLASSES}

    # Country-specific calibration floors. AO is allowed only through curated
    # AO-specific tokens; PT remains soft and has no floor.
    if any(item["specific"] == "br" and item["token"] in CURATED_BR_GIVEN_NAMES for item in evidence):
        smoothed["br"] = max(smoothed["br"], 4.5)
    if any(item["specific"] == "mz" and item["token"] in CURATED_MZ_SURNAMES for item in evidence):
        smoothed["mz"] = max(smoothed["mz"], 7.0)
    if sum(1 for item in evidence if item["specific"] == "cv") >= 2:
        smoothed["cv"] = max(smoothed["cv"], 7.0)
    elif any(item["specific"] == "cv" for item in evidence):
        smoothed["cv"] = max(smoothed["cv"], 5.0)
    if any(item["specific"] == "ao" and item["token"] in CURATED_AO_TOKENS for item in evidence):
        smoothed["ao"] = max(smoothed["ao"], 2.5)

    probs = _normalize_country_probs(smoothed)
    top_country, top_prob = _ranked_country_weights(probs)[0]
    if top_prob > 0.70 and top_country not in specific_countries:
        excess = top_prob - 0.70
        probs[top_country] = 0.70
        for country in _COUNTRY_CLASSES:
            if country != top_country:
                probs[country] += excess * prior[country] / (1.0 - prior[top_country])
        probs = _normalize_country_probs(probs)

    entropy_value = _country_entropy(probs)
    reasons = []
    for item in evidence:
        if item.get("specific"):
            token = item["token"]
            country = item["specific"]
            role = item["role"]
            reasons.append(f"{token}: {country.upper()} country-specific {role} signal")
    return probs, entropy_value, reasons


def _combine_country_attributions(
    first: ClassificationResult,
    second: ClassificationResult,
) -> tuple[dict[str, float], float, list[str]]:
    if not first.country_probs:
        return second.country_probs, second.country_entropy, []
    if not second.country_probs:
        return first.country_probs, first.country_entropy, []

    first_top, first_prob = _ranked_country_weights(first.country_probs)[0]
    second_top, second_prob = _ranked_country_weights(second.country_probs)[0]
    combined = {
        c: (first.country_probs.get(c, 0.0) * max(first.score, 1) + second.country_probs.get(c, 0.0) * max(second.score, 1))
        for c in _COUNTRY_CLASSES
    }
    reasons = []
    if first_top != second_top and first_prob >= 0.55 and second_prob >= 0.55:
        prior = _country_prior()
        combined = {c: combined[c] * 0.65 + prior[c] * 0.35 * (first.score + second.score) for c in _COUNTRY_CLASSES}
        reasons.append("joint_owner_country_conflict: flattened country attribution")

    probs = _normalize_country_probs(combined)
    return probs, _country_entropy(probs), reasons


def _extract_features(name: str) -> dict[str, float]:
    words = name.split()
    fn_table = _freq_tables.get("first_names", {})
    sn_table = _freq_tables.get("surnames", {})

    max_sn_br = max_fn_br = max_sn_hisp = max_fn_hisp = 0.0
    sn_in_census = fn_in_census = False
    surname_slots, firstname_slots = _role_slots(words)
    role_sn_br = role_fn_br = role_sn_hisp = role_fn_hisp = role_fn_american = 0.0
    definitive_hispanic_surname_count = 0
    portuguese_shared_surname_count = 0

    for i, w in enumerate(words):
        if w in ALL_PT_PREPS or _is_generational_suffix(words, i):
            continue
        sn = sn_table.get(w)
        if sn:
            sn_in_census = True
            max_sn_br = max(max_sn_br, sn.get("brazilian", 0))
            max_sn_hisp = max(max_sn_hisp, sn.get("hispanic", 0))
        fn = fn_table.get(w)
        if fn:
            fn_in_census = True
            max_fn_br = max(max_fn_br, fn.get("brazilian", 0))
            max_fn_hisp = max(max_fn_hisp, fn.get("hispanic", 0))

    for w in surname_slots:
        sn = sn_table.get(w)
        if sn:
            role_sn_br = max(role_sn_br, sn.get("brazilian", 0))
            role_sn_hisp = max(role_sn_hisp, sn.get("hispanic", 0))
            if w in _PORTUGUESE_SHARED_HISPANIC_SURNAMES:
                portuguese_shared_surname_count += 1
            elif sn.get("hispanic", 0) >= 0.72 and sn.get("brazilian", 0) < 0.35:
                definitive_hispanic_surname_count += 1

    for w in firstname_slots:
        fn = fn_table.get(w)
        if fn:
            role_fn_br = max(role_fn_br, fn.get("brazilian", 0))
            role_fn_hisp = max(role_fn_hisp, fn.get("hispanic", 0))
            role_fn_american = max(role_fn_american, fn.get("american", 0))

    ngram_probs = _ngram_pipeline.predict_proba([name])[0]
    classes = list(_ngram_pipeline.classes_)
    ngram_lusophone_prob = (
        float(ngram_probs[classes.index("lusophone")])
        if "lusophone" in classes
        else float(ngram_probs[classes.index("brazilian")])
    )

    valid_prep_positions = [
        i for i, w in enumerate(words)
        if w in ALL_PT_PREPS
        and any(words[j] not in ALL_PT_PREPS and not _is_generational_suffix(words, j) for j in range(i + 1, len(words)))
    ]
    pt_count = sum(1 for i in valid_prep_positions if words[i] in PT_STRONG_PREPS)
    has_de = any(words[i] in PT_WEAK_PREPS for i in valid_prep_positions)
    has_es = _has_spanish_preposition(words)
    if has_de and not has_es:
        pt_count += 1

    pcthisp = max((_us_census_pcthisp.get(w, 0) for w in words), default=0) / 100.0
    br_surname_with_nonbr_firstname = (
        1.0
        if role_sn_br >= 0.6 and role_fn_br < 0.25 and role_fn_american >= 0.2
        else 0.0
    )
    first_word = words[0] if words else ""
    last_word = next((words[i] for i in range(len(words) - 1, -1, -1) if words[i] not in ALL_PT_PREPS and not _is_generational_suffix(words, i)), "")
    first_as_surname = sn_table.get(first_word, {})
    first_as_firstname = fn_table.get(first_word, {})
    last_as_surname = sn_table.get(last_word, {})
    last_as_firstname = fn_table.get(last_word, {})
    last_first_br_signal = first_as_surname.get("brazilian", 0) * last_as_firstname.get("brazilian", 0)
    first_last_br_signal = (
        first_as_firstname.get("brazilian", 0) * last_as_surname.get("brazilian", 0)
        if first_as_surname.get("brazilian", 0) < 0.5
        else 0.0
    )
    first_last_hispanic_signal = first_as_firstname.get("hispanic", 0) * last_as_surname.get("hispanic", 0)
    pt_whitelist_membership = 0.0
    pt_given_slot_membership = 0.0
    country_specific_token_count = 0
    pt_surname_score = 0.0
    pt_surname_table_hit = 0.0
    lusophone_surname_score = 0.0
    lusophone_es_surname_count = 0
    for token in {token for token in surname_slots + firstname_slots if len(token) > 1}:
        given_entry = (_country_tables or {}).get("given_names", {}).get(token, {})
        given_weights = given_entry.get("combined_weights") or given_entry.get("weights") or {}
        if given_weights.get("pt", 0.0) >= 0.50:
            pt_whitelist_membership = 1.0
            if token in set(firstname_slots):
                pt_given_slot_membership = 1.0
        for section, role in (("given_names", "given"), ("surnames", "surname")):
            entry = (_country_tables or {}).get(section, {}).get(token)
            if not entry:
                continue
            weights = _normalize_country_evidence_weights(entry.get("combined_weights") or entry.get("weights") or {})
            if _is_country_specific(weights, token, role):
                country_specific_token_count += 1
    content_tokens = [
        w for i, w in enumerate(words)
        if w not in ALL_PT_PREPS and not _is_generational_suffix(words, i)
    ]
    for token in {token for token in surname_slots if len(token) > 1}:
        if token in CURATED_PT_GIVEN_NAMES:
            continue
        surname_entry = (_country_tables or {}).get("surnames", {}).get(token)
        if not surname_entry:
            continue
        surname_weights = _normalize_country_probs(
            surname_entry.get("combined_weights") or surname_entry.get("weights") or {}
        )
        pt_surname_score = max(pt_surname_score, surname_weights.get("pt", 0.0))
        if (surname_entry.get("counts") or {}).get("pt", 0) >= 10 and surname_weights.get("pt", 0.0) >= 0.10:
            pt_surname_table_hit = 1.0
        lusophone_surname_score = max(lusophone_surname_score, max(surname_weights.values(), default=0.0))
        if token.endswith("ES") and not token.endswith("EZ"):
            lusophone_es_surname_count += 1
    cv_cluster_score = 0.0
    mz_cluster_score = 0.0
    ao_cluster_score = 0.0
    for token in content_tokens:
        surname_entry = (_country_tables or {}).get("surnames", {}).get(token, {})
        weights = _normalize_country_evidence_weights(
            surname_entry.get("combined_weights") or surname_entry.get("weights") or {}
        )
        if token in CURATED_CV_SURNAMES and weights.get("cv", 0.0) >= 0.45 and weights.get("hispanic", 0.0) < 0.35 and weights.get("br", 0.0) < 0.60:
            cv_cluster_score += 1.0
        if token in CURATED_MZ_SURNAMES and weights.get("mz", 0.0) >= 0.45 and weights.get("hispanic", 0.0) < 0.35:
            mz_cluster_score += 1.0
        if token in CURATED_AO_TOKENS and weights.get("ao", 0.0) >= 0.45 and weights.get("hispanic", 0.0) < 0.35:
            ao_cluster_score += 1.0

    return {
        "max_surname_br_prob": max_sn_br,
        "max_firstname_br_prob": max_fn_br,
        "max_surname_hispanic_prob": max_sn_hisp,
        "max_firstname_hispanic_prob": max_fn_hisp,
        "role_surname_br_prob": role_sn_br,
        "role_firstname_br_prob": role_fn_br,
        "role_surname_hispanic_prob": role_sn_hisp,
        "role_firstname_hispanic_prob": role_fn_hisp,
        "role_firstname_american_prob": role_fn_american,
        "br_surname_with_nonbr_firstname": br_surname_with_nonbr_firstname,
        "last_first_br_signal": last_first_br_signal,
        "first_last_br_signal": first_last_br_signal,
        "first_last_hispanic_signal": first_last_hispanic_signal,
        "ngram_br_prob": ngram_lusophone_prob,
        "ngram_hispanic_prob": float(ngram_probs[classes.index("hispanic")]),
        "ngram_american_prob": float(ngram_probs[classes.index("american")]),
        "portuguese_preposition_count": pt_count,
        "spanish_preposition_present": 1.0 if has_es else 0.0,
        "brazilian_suffix_present": 1.0 if any(_is_generational_suffix(words, i) for i in range(len(words))) else 0.0,
        "ez_surname_count": sum(1 for w in words if w.endswith("EZ") and len(w) > 3),
        "merged_preposition_detected": 1.0 if any(w in MERGED_PREPS for w in words) else 0.0,
        "us_census_pcthispanic": pcthisp,
        "definitive_hispanic_surname_count": float(definitive_hispanic_surname_count),
        "portuguese_shared_surname_count": float(portuguese_shared_surname_count),
        "surname_in_any_census": 1.0 if sn_in_census else 0.0,
        "firstname_in_any_census": 1.0 if fn_in_census else 0.0,
        "compound_name_detected": 1.0 if _has_compound_first_name(words) else 0.0,
        "name_token_count": len(words),
        "both_names_have_signal": 0.0,
        "pt_whitelist_membership": pt_whitelist_membership,
        "pt_given_slot_membership": pt_given_slot_membership,
        "cv_cluster_score": cv_cluster_score,
        "mz_cluster_score": mz_cluster_score,
        "ao_cluster_score": ao_cluster_score,
        "country_specific_token_count": float(country_specific_token_count),
        "pt_surname_score": pt_surname_score,
        "pt_surname_table_hit": pt_surname_table_hit,
        "lusophone_surname_score": lusophone_surname_score,
        "lusophone_es_surname_count": float(lusophone_es_surname_count),
    }


def _meta_predict(features: dict[str, float]) -> float:
    x = np.array([features.get(f, 0) for f in FEATURE_NAMES])
    z = float(np.dot(_meta_coef, x) + _meta_intercept)
    return 1.0 / (1.0 + math.exp(-z))


def _surname_origin_floor(name: str, features: dict[str, float]) -> tuple[int, str] | None:
    """Give Anglicized Portuguese/Brazilian surname leads visible low confidence.

    Thresholds are calibrated to the honest US-bearer frequency tables, where
    even distinctly Brazilian surnames rarely exceed ~0.7 P(BR) and shared
    Iberian surnames sit at 0.3-0.5."""
    if features["spanish_preposition_present"] or features["ez_surname_count"]:
        return None
    if features["max_surname_hispanic_prob"] >= 0.70:
        return None

    words = name.split()
    fn_table = _freq_tables.get("first_names", {})
    sn_table = _freq_tables.get("surnames", {})
    first_word = words[0] if words else ""
    last_word = next((words[i] for i in range(len(words) - 1, -1, -1) if words[i] not in ALL_PT_PREPS and not _is_generational_suffix(words, i)), "")
    first_fn = fn_table.get(first_word, {})
    last_sn = sn_table.get(last_word, {})

    last_first_signal = (
        features["role_surname_br_prob"] >= 0.35
        and features["role_surname_hispanic_prob"] <= 0.70
        and features["role_firstname_american_prob"] >= 0.60
        and features["role_firstname_br_prob"] < 0.25
    )
    first_last_signal = (
        first_fn.get("american", 0) >= 0.60
        and last_sn.get("brazilian", 0) >= 0.35
        and last_sn.get("hispanic", 0) <= 0.70
    )

    if not (last_first_signal or first_last_signal):
        return None

    strong_sn = max(features["role_surname_br_prob"], last_sn.get("brazilian", 0))
    if features["ngram_br_prob"] >= 0.35 or strong_sn >= 0.60:
        return 30, "surname_origin_lead:strong"
    return 24, "surname_origin_lead"


def _double_br_surname_floor(name: str, features: dict[str, float]) -> tuple[int, str] | None:
    """Two distinctly Brazilian surnames are strong evidence even when the
    given name is Anglo (HUDSON, KEVIN are popular in Brazil)."""
    if features["spanish_preposition_present"] or features["ez_surname_count"]:
        return None
    words = name.split()
    sn_table = _freq_tables.get("surnames", {})
    br_surnames = 0
    for i, w in enumerate(words):
        if w in ALL_PT_PREPS or _is_generational_suffix(words, i):
            continue
        sn = sn_table.get(w)
        if sn and sn.get("brazilian", 0) >= 0.60 and sn.get("hispanic", 0) <= 0.35:
            br_surnames += 1
    if br_surnames >= 2:
        return 60, "double_br_surname_floor"
    return None


def _lusophone_country_floor(features: dict[str, float]) -> tuple[int, str] | None:
    if features["spanish_preposition_present"] or features["ez_surname_count"]:
        return None
    if features["mz_cluster_score"] >= 1:
        return 50, "lusophone_mz_cluster_floor"
    if features["cv_cluster_score"] >= 2:
        return 50, "lusophone_cv_multi_cluster_floor"
    if features["cv_cluster_score"] >= 1:
        return 30, "lusophone_cv_cluster_floor"
    if features["ao_cluster_score"] >= 1:
        return 30, "lusophone_ao_cluster_floor"
    if features.get("pt_given_slot_membership", 0.0) and features.get("pt_surname_table_hit", 0.0):
        return 15, "lusophone_pt_whitelist_surname_floor"
    if features["pt_whitelist_membership"] and features["ngram_br_prob"] >= 0.70:
        return 30, "lusophone_pt_soft_floor"
    if features["ngram_br_prob"] >= 0.88 and features["max_surname_hispanic_prob"] < 0.70:
        return 15, "lusophone_ngram_floor"
    return None


def _merged_preposition_floor(features: dict[str, float]) -> tuple[int, str] | None:
    if (
        features["merged_preposition_detected"]
        and features["role_firstname_american_prob"] >= 0.60
        and features["role_firstname_br_prob"] < 0.25
    ):
        return 30, "merged_surname_origin_floor"
    return None


def _merged_anglo_cap(features: dict[str, float]) -> tuple[int, str] | None:
    if (
        features["merged_preposition_detected"]
        and features["role_firstname_american_prob"] >= 0.60
        and features["role_firstname_br_prob"] < 0.25
    ):
        return 45, "merged_surname_origin_cap"
    return None


def _dangling_prep_unknown_given_cap(name: str, features: dict[str, float]) -> tuple[int, str] | None:
    words = name.split()
    if (
        words
        and words[-1] in ALL_PT_PREPS
        and features["role_firstname_br_prob"] == 0
        and not features["compound_name_detected"]
        and not features["merged_preposition_detected"]
        and not features["brazilian_suffix_present"]
    ):
        return 49, "dangling_preposition_unknown_given_cap"
    return None


def _has_lusophone_specific_evidence(name: str, features: dict[str, float]) -> bool:
    words = name.split()
    if any(w in PT_STRONG_PREPS for w in words):
        return True
    if (
        features["merged_preposition_detected"]
        or features["brazilian_suffix_present"]
        or features["compound_name_detected"]
        or features["cv_cluster_score"] >= 1
        or features["mz_cluster_score"] >= 1
        or features["ao_cluster_score"] >= 1
        or features.get("pt_surname_table_hit", 0.0)
        or features.get("country_specific_token_count", 0.0) >= 1
    ):
        return True
    if features["role_surname_br_prob"] >= 0.60 and features["role_surname_hispanic_prob"] <= 0.35:
        return True
    if features["role_surname_br_prob"] >= 0.45 and features["role_surname_hispanic_prob"] <= 0.55:
        return True
    if features["max_surname_br_prob"] >= 0.35 and features["max_surname_hispanic_prob"] <= 0.60:
        return True
    if features["max_firstname_br_prob"] >= 0.60 and features["max_firstname_hispanic_prob"] <= 0.60:
        return True
    if (
        features["ngram_br_prob"] >= 0.90
        and features["max_surname_hispanic_prob"] < 0.75
        and (features["max_surname_br_prob"] >= 0.30 or features["max_firstname_br_prob"] >= 0.55)
    ):
        return True
    if features["max_firstname_br_prob"] >= 0.75 and features["max_firstname_hispanic_prob"] <= 0.35:
        return True
    if any(w.endswith(("EIRO", "EIRA")) and len(w) > 5 for w in words):
        return True
    if any(w in CURATED_BR_GIVEN_NAMES for w in words):
        return True
    return False


def _has_structural_lusophone_evidence(name: str, features: dict[str, float]) -> bool:
    words = name.split()
    if any(w in PT_STRONG_PREPS for w in words):
        return True
    if (
        features["merged_preposition_detected"]
        or features["brazilian_suffix_present"]
        or features["cv_cluster_score"] >= 1
        or features["mz_cluster_score"] >= 1
        or features["ao_cluster_score"] >= 1
        or features.get("pt_surname_table_hit", 0.0)
    ):
        return True
    if features["role_surname_br_prob"] >= 0.60 and features["role_surname_hispanic_prob"] <= 0.35:
        return True
    if any(w.endswith(("EIRO", "EIRA")) and len(w) > 5 for w in words):
        return True
    return False


def _has_definitive_hispanic_evidence(features: dict[str, float]) -> bool:
    return (
        features["ez_surname_count"] > 0
        or features["spanish_preposition_present"]
        or features.get("definitive_hispanic_surname_count", 0.0) >= 1
        or (
            features["us_census_pcthispanic"] >= 0.72
            and features["role_surname_br_prob"] < 0.35
            and features.get("portuguese_shared_surname_count", 0.0) == 0
        )
    )


def _haitian_french_guard_cap(name: str, features: dict[str, float]) -> tuple[int, str] | None:
    words = set(name.replace("-", " ").split())
    if not (
        words & _HAITIAN_FRENCH_TOKENS
        or any(w.endswith("CIUS") or w.startswith("SAINT") for w in words)
    ):
        return None
    return 10, "haitian_french_guard_cap"


def _definitive_hispanic_surname_cap(name: str, features: dict[str, float]) -> tuple[int, str] | None:
    if _has_structural_lusophone_evidence(name, features):
        return None
    if features.get("definitive_hispanic_surname_count", 0.0) >= 1:
        return 40, "definitive_hispanic_surname_cap"
    if (
        features["us_census_pcthispanic"] >= 0.72
        and features["role_surname_br_prob"] < 0.35
        and features.get("portuguese_shared_surname_count", 0.0) == 0
    ):
        return 40, "pcthispanic_surname_cap"
    return None


def _very_high_specific_evidence_cap(name: str, features: dict[str, float]) -> tuple[int, str] | None:
    if _has_lusophone_specific_evidence(name, features):
        return None
    return 69, "very_high_requires_lusophone_specific_evidence"


def _hispanic_evidence_cap(features: dict[str, float]) -> tuple[int, str] | None:
    """Cap names carrying distinctly Spanish evidence with no Portuguese
    counter-evidence (prepositions, merged forms, suffixes)."""
    if features["spanish_preposition_present"] or features["ez_surname_count"]:
        if features["ngram_hispanic_prob"] >= 0.70:
            return 10, "hispanic_marker_strong_cap"
        return 40, "hispanic_marker_cap"
    has_pt_evidence = (
        features["portuguese_preposition_count"] > 0
        or features["merged_preposition_detected"]
        or features["brazilian_suffix_present"]
        or features.get("lusophone_es_surname_count", 0) > 0
    )
    if has_pt_evidence:
        return None
    if (
        features["max_surname_hispanic_prob"] >= 0.85
        and features["max_firstname_hispanic_prob"] >= 0.85
        and features["max_firstname_br_prob"] < 0.50
    ):
        if features["ngram_hispanic_prob"] >= 0.60 or features["us_census_pcthispanic"] >= 0.70:
            return 10, "hispanic_name_strong_cap"
        return 15, "hispanic_name_dominant_cap"
    if (
        features["max_surname_hispanic_prob"] >= 0.85
        and features["max_firstname_br_prob"] < 0.50
    ):
        return 20, "hispanic_surname_dominant_cap"
    if (
        features["max_firstname_hispanic_prob"] >= 0.85
        and features["max_firstname_br_prob"] < 0.50
    ):
        return 40, "hispanic_first_name_cap"
    if (
        features["us_census_pcthispanic"] >= 0.60
        and features["max_firstname_br_prob"] < 0.60
        and features["max_surname_br_prob"] < 0.50
        and not features["compound_name_detected"]
    ):
        return 40, "hispanic_surname_majority_cap"
    return None


def _no_brazilian_evidence_cap(name: str, features: dict[str, float]) -> tuple[int, str] | None:
    """The meta-model's priors can inflate names with no Brazilian signal at
    all (e.g. unknown hyphenated surnames). If nothing points to Brazil,
    the score cannot exceed noise level. Only STRONG Portuguese prepositions
    (DA/DOS/DAS/DO) exempt — a bare DE also appears in Flemish/French/Italian
    names (VAN MARCKE DE LUMMEN)."""
    has_strong_prep = any(w in PT_STRONG_PREPS for w in name.split())
    if (
        has_strong_prep
        or features["merged_preposition_detected"]
        or features["brazilian_suffix_present"]
        or features["compound_name_detected"]
        or features["max_surname_br_prob"] >= 0.35
        or features["cv_cluster_score"] >= 1
        or features["mz_cluster_score"] >= 1
        or features["ao_cluster_score"] >= 1
        or (features.get("pt_given_slot_membership", 0.0) and features.get("pt_surname_table_hit", 0.0))
    ):
        return None
    # A surname that IS in the census tables with ~zero Brazilian share
    # (WANG, VU, KUMAR, SNIPES) positively identifies a non-Brazilian family;
    # neither the char model nor a shared given name can override it.
    # Role-based: judged on the surname-slot token, so a Brazilian given
    # name that doubles as a surname elsewhere (RUI) cannot mask it.
    known_foreign_surname = (
        features["surname_in_any_census"] and features["role_surname_br_prob"] < 0.05
    )
    if features["max_firstname_br_prob"] < 0.50:
        # No census signal at all: out-of-distribution names (Chinese,
        # Indian, Slavic...) can still get a high ngram_br_prob, so the
        # ngram alone is not trusted below near-certainty.
        if known_foreign_surname or features["ngram_br_prob"] < 0.75:
            return 10, "no_brazilian_evidence_cap"
        return None
    # Single Brazilian-leaning first name with no surname corroboration
    # (KARINE, LILIAN, MARCIA, RUI are also Haitian/European/Chinese) —
    # visible at Low/Medium, never Very High. A DISTINCTLY Brazilian given
    # name (THIAGO-grade, >=0.70) is trusted on its own.
    if known_foreign_surname:
        if (
            features["us_census_pcthispanic"] >= 0.65
            or features["max_surname_hispanic_prob"] >= 0.85
            or features["ngram_hispanic_prob"] >= 0.55
        ):
            return 14, "hispanic_foreign_surname_cap"
        return 30, "foreign_surname_cap"
    if features["max_firstname_br_prob"] >= 0.70:
        return None
    if features["ngram_br_prob"] < 0.75:
        return 49, "single_first_name_only_cap"
    return None


def _single_token_cap(features: dict[str, float]) -> tuple[int, str] | None:
    """A one-word name is never strong evidence by itself."""
    if features["name_token_count"] != 1 or features["merged_preposition_detected"]:
        return None
    if features["max_firstname_br_prob"] >= 0.60:
        return 30, "single_token_given_name_cap"
    if features["ngram_br_prob"] < 0.60:
        return 10, "single_token_cap"
    # Even a Brazilian-looking single unknown token is at most Medium
    return 49, "single_token_cap"


def _american_dominant_cap(features: dict[str, float]) -> tuple[int, str] | None:
    if (
        features["ngram_american_prob"] >= 0.80
        and features["role_firstname_american_prob"] >= 0.80
        and features["portuguese_preposition_count"] == 0
        and not features["merged_preposition_detected"]
        and not features["brazilian_suffix_present"]
    ):
        return 10, "american_dominant_cap"
    return None


def _has_hispanic_evidence(features: dict[str, float]) -> bool:
    """Strong Hispanic signal in a name — used as negative evidence when it
    appears in the co-owner of a property record. Thresholds are strict so
    ambiguous shared-Iberian co-owners do not penalize Brazilian records:
    a shared surname (CASTRO, CRUZ) only counts when the given name is not
    Brazilian-leaning, but hard markers (-EZ, Spanish prepositions) always do."""
    if _has_definitive_hispanic_evidence(features):
        return True
    if features["max_firstname_br_prob"] >= 0.50:
        return False
    return (
        features["role_surname_hispanic_prob"] >= 0.85
        or features["max_firstname_hispanic_prob"] >= 0.70
        or features["ngram_hispanic_prob"] >= 0.75
    )


def _confidence_label(score: int) -> str:
    for level, threshold in CONFIDENCE_THRESHOLDS.items():
        if score >= threshold:
            return level
    return "Very Low"


def _build_reasons(features: dict[str, float], name: str) -> list[str]:
    reasons = []
    words = name.split()
    fn_table = _freq_tables.get("first_names", {})
    sn_table = _freq_tables.get("surnames", {})

    for i, w in enumerate(words):
        sn = sn_table.get(w)
        if sn and sn.get("brazilian", 0) > 0.3:
            reasons.append(f"{w}: {int(sn['brazilian'] * 100)}% BR surname (census)")
    for i, w in enumerate(words):
        if w in ALL_PT_PREPS or _is_generational_suffix(words, i):
            continue
        fn = fn_table.get(w)
        sn = sn_table.get(w)
        if fn and fn.get("brazilian", 0) > 0.5 and not (sn and sn.get("brazilian", 0) > 0.5):
            reasons.append(f"{w}: {int(fn['brazilian'] * 100)}% BR first name (census)")
    if features["ngram_br_prob"] > 0.5:
        reasons.append(f"Name pattern: {int(features['ngram_br_prob'] * 100)}% BR (character model)")
    for i, w in enumerate(words):
        if w in PT_STRONG_PREPS:
            reasons.append(f"{w}: Portuguese preposition")
        if _is_generational_suffix(words, i):
            reasons.append(f"{w}: Brazilian generational suffix")
    if features["merged_preposition_detected"]:
        reasons.append("Merged preposition detected (e.g., DASILVA)")
    if features["compound_name_detected"]:
        reasons.append("Brazilian compound first name")
    return reasons


def _classify_name_with_features(name: str) -> tuple[ClassificationResult, dict[str, float] | None]:
    """Internal: classify and also return the feature vector (None for
    empty/business names) so record-level scoring can weigh co-owner evidence."""
    _load_models()
    name, is_business = _normalize_name(name)
    if not name or is_business:
        return ClassificationResult(), None

    features = _extract_features(name)
    prob = _meta_predict(features)
    score = min(100, max(0, round(prob * 100)))
    calibration_reason = None
    floor = _surname_origin_floor(name, features)
    double_floor = _double_br_surname_floor(name, features)
    if double_floor and (not floor or double_floor[0] > floor[0]):
        floor = double_floor
    lusophone_floor = _lusophone_country_floor(features)
    if lusophone_floor and (not floor or lusophone_floor[0] > floor[0]):
        floor = lusophone_floor
    merged_floor = _merged_preposition_floor(features)
    if merged_floor and (not floor or merged_floor[0] > floor[0]):
        floor = merged_floor
    if floor:
        floor_score, calibration_reason = floor
        score = max(score, floor_score)
    cap = _merged_anglo_cap(features)
    haitian_cap = _haitian_french_guard_cap(name, features)
    if haitian_cap and (not cap or haitian_cap[0] < cap[0]):
        cap = haitian_cap
    dangling_cap = _dangling_prep_unknown_given_cap(name, features)
    if dangling_cap and (not cap or dangling_cap[0] < cap[0]):
        cap = dangling_cap
    definitive_hispanic_cap = _definitive_hispanic_surname_cap(name, features)
    if definitive_hispanic_cap and (not cap or definitive_hispanic_cap[0] < cap[0]):
        cap = definitive_hispanic_cap
    hispanic_cap = _hispanic_evidence_cap(features)
    if hispanic_cap and (not cap or hispanic_cap[0] < cap[0]):
        cap = hispanic_cap
    single_cap = _single_token_cap(features)
    if single_cap and (not cap or single_cap[0] < cap[0]):
        cap = single_cap
    no_br_cap = _no_brazilian_evidence_cap(name, features)
    if no_br_cap and (not cap or no_br_cap[0] < cap[0]):
        cap = no_br_cap
    american_cap = _american_dominant_cap(features)
    if american_cap and (not cap or american_cap[0] < cap[0]):
        cap = american_cap
    vh_cap = _very_high_specific_evidence_cap(name, features)
    if vh_cap and (not cap or vh_cap[0] < cap[0]):
        cap = vh_cap
    if cap:
        cap_score, cap_reason = cap
        if score > cap_score:
            score = cap_score
            calibration_reason = cap_reason
    confidence = _confidence_label(score)
    reasons = _build_reasons(features, name) if score >= 15 else []
    if calibration_reason:
        reasons.append(calibration_reason)
    country_probs, country_entropy, country_reasons = _country_attribution_for_name(name)
    if score >= 15:
        reasons.extend(country_reasons)

    result = ClassificationResult(
        score=score,
        confidence=confidence,
        reasons=reasons,
        probabilities={
            "brazilian": round(prob, 3),
            "hispanic": round(features["ngram_hispanic_prob"], 3),
            "american": round(features["ngram_american_prob"], 3),
        },
        country_probs={country: round(prob, 4) for country, prob in country_probs.items()},
        country_entropy=round(country_entropy, 4),
    )
    return result, features


def classify_name(name: str) -> ClassificationResult:
    """Classify a single name string. Returns score 0-100 with reasons."""
    result, _ = _classify_name_with_features(name)
    return result


def classify_record(name1: str, name2: str = "") -> ClassificationResult:
    """Classify a property record with two owner name fields.

    A co-owner with strong Hispanic evidence and no Brazilian signal is
    NEGATIVE evidence: the household is most likely Hispanic, so the record
    is penalized and capped below Very High instead of inheriting the
    higher name's score unchanged."""
    r1, f1 = _classify_name_with_features(name1)
    r2, f2 = _classify_name_with_features(name2) if name2 else (ClassificationResult(), None)

    (base, base_f), (other, other_f) = ((r1, f1), (r2, f2)) if r1.score >= r2.score else ((r2, f2), (r1, f1))

    other_is_hispanic = other_f is not None and _has_hispanic_evidence(other_f)
    base_is_hispanic = base_f is not None and _has_hispanic_evidence(base_f)

    reasons = list(base.reasons)
    if base_is_hispanic and not (
        base_f and _has_structural_lusophone_evidence(name1 if base is r1 else name2, base_f)
    ):
        final_score = min(base.score, 49)
        reasons.append("Owner name shows definitive Hispanic surname indicators")
    elif other_is_hispanic:
        final_score = max(0, min(base.score - 20, 49))
        reasons.append("Co-owner name shows Hispanic indicators")
    else:
        bonus = min(10, (base.score + other.score) // 10) if base.score > 0 and other.score > 0 else 0
        final_score = min(100, base.score + bonus)
        if other.reasons:
            reasons.append("Both owner names show Brazilian indicators")
    country_probs, country_entropy, country_reasons = _combine_country_attributions(r1, r2)
    reasons.extend(country_reasons)

    return ClassificationResult(
        score=final_score,
        confidence=_confidence_label(final_score),
        reasons=reasons,
        probabilities=base.probabilities,
        country_probs={country: round(prob, 4) for country, prob in country_probs.items()},
        country_entropy=round(country_entropy, 4),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="*", help="Name to classify.")
    parser.add_argument("--explain", action="store_true", help="Print probabilities, country attribution, and reasons.")
    args = parser.parse_args()

    if args.name:
        result = classify_name(" ".join(args.name))
        print(f"Score: {result.score}/100 ({result.confidence})")
        if args.explain:
            print(f"Probabilities: {result.probabilities}")
            print(f"Country probabilities: {result.country_probs}")
            print(f"Country entropy: {result.country_entropy}")
            for r in result.reasons:
                print(f"  - {r}")
    else:
        tests = [
            ("FERREIRA GUSTAVO DA SILVA", "High+"),
            ("GENIVALDO DE SOUZA", "High+"), ("CLEUDIMAR SANTOS", "High+"),
            ("EDILEUSA OLIVEIRA", "High+"), ("DASILVA THIAGO", "High+"),
            ("TEIXEIRA ANA PAULA", "High+"),
            ("NASCIMENTO JOAO PEDRO JUNIOR", "High+"),
            ("GIVANILDO PEREIRA", "High+"), ("ROSANGELA DOS SANTOS", "High+"),
            ("WANDERLEI SILVA", "High+"),
            ("HERNANDEZ JUAN CARLOS", "Low-"), ("GONZALEZ MARIA DE LOS ANGELES", "Low-"),
            ("MARTINEZ GUADALUPE", "Low-"), ("SMITH JOHN", "Low-"),
            ("JOHNSON MICHAEL", "Low-"), ("WILLIAMS JENNIFER", "Low-"),
            ("GARCIA JOSE", "Low-"), ("GARCIA DA SILVA JOSE", "High+"),
            ("SILVA PROPERTIES LLC", "Low-"), ("COSTA MARIA", "Low-"),
        ]
        passed = 0
        for name, expected in tests:
            r = classify_name(name)
            thresh = CONFIDENCE_THRESHOLDS.get(expected.rstrip("+-"), 0)
            ok = r.score >= thresh if expected.endswith("+") else r.score < thresh
            passed += ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:45s} score={r.score:3d} ({r.confidence:10s})")
        print(f"\n  {passed}/{len(tests)} tests passed")
