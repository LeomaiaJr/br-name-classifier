"""Extract the feature vector for the meta-classifier."""

import json
import pickle
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, MODELS_DIR
from constants import (
    PT_STRONG_PREPS, PT_WEAK_PREPS, ALL_PT_PREPS, ES_PATTERNS,
    BR_SUFFIXES, MERGED_PREPS, COMPOUND_FIRST_NAMES, BUSINESS_KEYWORDS,
    FEATURE_NAMES, CURATED_PT_GIVEN_NAMES, CURATED_CV_SURNAMES,
    CURATED_AO_TOKENS, CURATED_MZ_SURNAMES,
)

_freq_tables = None
_ngram_pipeline = None
_us_census_pcthisp = None
_country_tables = None
_SPANISH_PREP_SEQS = tuple(tuple(p.split()) for p in ES_PATTERNS)
_COMPOUND_FIRST_NAME_SEQS = tuple(tuple(cn.split()) for cn in COMPOUND_FIRST_NAMES)
_COUNTRY_CLASSES = ("br", "pt", "cv", "ao", "mz", "palop_other")
_SHARED_LUSOPHONE_TOKENS = frozenset({
    "SILVA", "SANTOS", "PEREIRA", "FERREIRA", "GOMES", "LOPES",
    "RODRIGUES", "COSTA", "MARTINS", "FERNANDES", "JOSE", "MARIA",
    "JOAO", "ANTONIO", "FRANCISCO", "ANA", "PEDRO", "MANUEL", "PAULO",
    "MIGUEL", "OLIVEIRA", "SOUSA", "SOUZA", "RIBEIRO",
})


def _load():
    global _freq_tables, _ngram_pipeline, _us_census_pcthisp, _country_tables
    if _freq_tables is None:
        with open(OUTPUT_DIR / "frequency_tables.json") as f:
            _freq_tables = json.load(f)
    if _country_tables is None:
        path = OUTPUT_DIR / "country_tables.json"
        _country_tables = json.loads(path.read_text()) if path.exists() else {}
    if _ngram_pipeline is None:
        with open(MODELS_DIR / "ngram_pipeline.pkl", "rb") as f:
            _ngram_pipeline = pickle.load(f)
    if _us_census_pcthisp is None:
        path = Path(__file__).parent.parent / "data" / "processed" / "all_surnames.csv"
        _us_census_pcthisp = {}
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df[df["country"] == "american"].iterrows():
                try:
                    pct = float(row["pcthispanic"])
                    if pct > 0:
                        _us_census_pcthisp[str(row["name"])] = pct
                except (ValueError, TypeError):
                    pass


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _has_sequence(words: list[str], seq: tuple[str, ...]) -> bool:
    if len(seq) > len(words):
        return False
    return any(tuple(words[i:i + len(seq)]) == seq for i in range(len(words) - len(seq) + 1))


def _has_spanish_preposition(words: list[str]) -> bool:
    return any(_has_sequence(words, seq) for seq in _SPANISH_PREP_SEQS)


def _has_compound_first_name(words: list[str]) -> bool:
    return any(_has_sequence(words, seq) for seq in _COMPOUND_FIRST_NAME_SEQS)


def _is_generational_suffix(words: list[str], index: int) -> bool:
    return words[index] in BR_SUFFIXES and index > 0


def _normalize_country_probs(probs: dict[str, float]) -> dict[str, float]:
    clean = {c: max(0.0, float(probs.get(c, 0.0))) for c in _COUNTRY_CLASSES}
    total = sum(clean.values())
    if total <= 0:
        return {c: 0.0 for c in _COUNTRY_CLASSES}
    return {c: clean[c] / total for c in _COUNTRY_CLASSES}


def _is_country_specific(weights: dict[str, float], token: str, role: str) -> str | None:
    ranked = sorted(
        ((c, float(weights.get(c, 0.0))) for c in _COUNTRY_CLASSES),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] <= 0:
        return None
    top_country, top_weight = ranked[0]
    second_weight = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_weight <= 2 * max(second_weight, 1e-9) or token in _SHARED_LUSOPHONE_TOKENS:
        return None
    if top_country == "ao":
        return "ao" if token in CURATED_AO_TOKENS else None
    if top_country == "pt":
        return "pt" if role == "given" and top_weight >= 0.60 else None
    if top_country == "br" and role != "given":
        return None
    if top_country == "cv" and role != "surname":
        return None
    if top_country == "mz" and role != "surname":
        return None
    return top_country


def _role_slots(words: list[str]) -> tuple[list[str], list[str]]:
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


def extract_features(name: str) -> dict:
    _load()
    name = strip_accents(name).upper().strip().replace(",", " ")
    name = " ".join(name.split())

    if not name or set(name.split()) & BUSINESS_KEYWORDS:
        return {f: 0.0 for f in FEATURE_NAMES}

    words = name.split()
    fn_table = _freq_tables.get("first_names", {})
    sn_table = _freq_tables.get("surnames", {})

    max_sn_br = max_fn_br = max_sn_hisp = max_fn_hisp = 0.0
    sn_in_census = fn_in_census = False
    surname_slots, firstname_slots = _role_slots(words)
    role_sn_br = role_fn_br = role_sn_hisp = role_fn_hisp = role_fn_american = 0.0

    for i, w in enumerate(words):
        if w in ALL_PT_PREPS or _is_generational_suffix(words, i):
            continue
        if w in sn_table:
            sn_in_census = True
            max_sn_br = max(max_sn_br, sn_table[w].get("brazilian", 0))
            max_sn_hisp = max(max_sn_hisp, sn_table[w].get("hispanic", 0))
        if w in fn_table:
            fn_in_census = True
            max_fn_br = max(max_fn_br, fn_table[w].get("brazilian", 0))
            max_fn_hisp = max(max_fn_hisp, fn_table[w].get("hispanic", 0))

    for w in surname_slots:
        if w in sn_table:
            role_sn_br = max(role_sn_br, sn_table[w].get("brazilian", 0))
            role_sn_hisp = max(role_sn_hisp, sn_table[w].get("hispanic", 0))

    for w in firstname_slots:
        if w in fn_table:
            role_fn_br = max(role_fn_br, fn_table[w].get("brazilian", 0))
            role_fn_hisp = max(role_fn_hisp, fn_table[w].get("hispanic", 0))
            role_fn_american = max(role_fn_american, fn_table[w].get("american", 0))

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
    has_es = _has_spanish_preposition(words)
    if any(words[i] in PT_WEAK_PREPS for i in valid_prep_positions) and not has_es:
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
            weights = _normalize_country_probs(entry.get("combined_weights") or entry.get("weights") or {})
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
    cv_cluster_score = float(sum(1 for token in content_tokens if token in CURATED_CV_SURNAMES))
    mz_cluster_score = float(sum(1 for token in content_tokens if token in CURATED_MZ_SURNAMES))
    ao_cluster_score = float(sum(1 for token in content_tokens if token in CURATED_AO_TOKENS))

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


def extract_features_batch(names: list[str]) -> pd.DataFrame:
    records = []
    for i, name in enumerate(names):
        records.append(extract_features(name))
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1:,}/{len(names):,}")
    return pd.DataFrame(records)


if __name__ == "__main__":
    for name in ["FERREIRA GUSTAVO DA SILVA", "GENIVALDO DE SOUZA", "SMITH JOHN", "HERNANDEZ JUAN CARLOS"]:
        feats = extract_features(name)
        print(f"\n{name}:")
        for k, v in feats.items():
            if v != 0:
                print(f"  {k}: {v}")
