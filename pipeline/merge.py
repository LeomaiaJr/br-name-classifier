"""Merge normalized census data into cross-country frequency tables."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import (
    PROJECT_ROOT, PROCESSED_DIR, OUTPUT_DIR,
    FREQUENCY_MIN_OCCURRENCES, PROBABILITY_DECIMALS, MODEL_VERSION,
    LUSOPHONE_CLASSES, LUSOPHONE_UNATTRIBUTABLE_PRIOR,
)
from constants import CURATED_LUSOPHONE_BOOSTS
from pipeline.normalize import clean_name, normalize_lusophone


COUNTRY_TABLE_CLASSES = tuple(LUSOPHONE_CLASSES) + ("hispanic",)
PROTECTED_COUNTRY_SPECIFIC_SURNAMES = {
    # High-specificity CV surnames approved in C4. Spanish-shared CV seeds
    # DELGADO/MORENO/SANCHES deliberately stay outside this set.
    "SEMEDO": "cv",
    "FORTES": "cv",
    "VARELA": "cv",
    "DE PINA": "cv",
    "DA LUZ": "cv",
    "EVORA": "cv",
    "FURTADO": "cv",
    # MZ and AO clusters are curated as country-specific evidence.
    "SITOE": "mz",
    "SITHOE": "mz",
    "MACAMO": "mz",
    "MONDLANE": "mz",
    "LANGA": "mz",
    "COSSA": "mz",
    "TEMBE": "mz",
    "MANJATE": "mz",
    "MACHAVA": "mz",
    "NHANTUMBO": "mz",
    "MACUACUA": "mz",
    "MUCHANGA": "mz",
    "MATSINHE": "mz",
    "MUIANGA": "mz",
    "MUNGUAMBE": "mz",
    "BANZE": "mz",
    "BILA": "mz",
    "MANHIQUE": "mz",
    "SIMBINE": "mz",
    "GUAMBE": "mz",
    "ZANDAMELA": "mz",
    "MANDLATE": "mz",
    "MAGAIA": "mz",
    "VILANCULOS": "mz",
    "FUMO": "mz",
    "MANHICA": "mz",
    "CUMBE": "mz",
    "KIALA": "ao",
    "BUNGA": "ao",
    "DALA": "ao",
    "KALUNGA": "ao",
    "KASSOMA": "ao",
    "CHIVUKUVUKU": "ao",
    "SAMAKUVA": "ao",
    "MAVINGA": "ao",
    "KANDIMBA": "ao",
    "KAPENDA": "ao",
}
PROTECTED_HOME_MIN_WEIGHT = 0.74


# Brazilian surnames — seeded from the existing curated lists in brazilian_names.py
# These get Brazilian frequency estimates since IBGE doesn't publish surname data
BRAZILIAN_SURNAME_SEEDS = {
    # Tier 1 - distinctly Brazilian (high frequency estimate)
    "FERREIRA": 2000000, "TEIXEIRA": 800000, "CARVALHO": 1500000, "NASCIMENTO": 900000,
    "GONCALVES": 800000, "NOGUEIRA": 400000, "COELHO": 350000, "PINHEIRO": 400000,
    "BEZERRA": 350000, "CAVALCANTE": 350000, "CAVALCANTI": 200000, "GUIMARAES": 400000,
    "VASCONCELOS": 250000, "FIGUEIREDO": 200000, "AZEVEDO": 400000, "QUEIROZ": 300000,
    "SIQUEIRA": 250000, "LACERDA": 150000, "MESQUITA": 200000, "PEIXOTO": 150000,
    "CUNHA": 500000, "CERQUEIRA": 150000, "CHAGAS": 200000, "CORDEIRO": 250000,
    "DINIZ": 200000, "DOMINGUES": 200000, "GALVAO": 150000, "GUEDES": 150000,
    "MAGALHAES": 350000, "MENEZES": 200000, "MORAES": 350000, "MORAIS": 250000,
    "NEVES": 300000, "NOBREGA": 100000, "PASSOS": 150000, "REZENDE": 200000,
    "SALGADO": 100000, "TAVARES": 300000, "VIANA": 250000, "CONCEICAO": 200000,
    "ASSUNCAO": 100000, "BITTENCOURT": 100000, "MASCARENHAS": 80000, "CAMARGO": 200000,
    "BARRETO": 200000, "BRANDAO": 200000, "DOURADO": 80000, "FONTES": 100000,
    "MOURA": 400000, "PIRES": 250000, "XAVIER": 300000, "ALENCAR": 150000,
    "CABRAL": 200000, "COUTINHO": 100000, "MACEDO": 200000, "LEMOS": 150000,
    "BASTOS": 150000, "FRAGA": 100000, "GODOY": 100000, "PADILHA": 100000,
    "REGO": 100000, "BISPO": 150000, "ANTUNES": 200000, "AMORIM": 200000,
    "MARINHO": 200000, "FREIRE": 150000, "MENDONCA": 200000, "FAGUNDES": 100000,
    "FERRAZ": 100000, "FURTADO": 100000, "FARIA": 250000, "CAETANO": 150000,
    "CHAVES": 200000, "GONZAGA": 80000, "FEITOSA": 150000,
    "BARROSO": 100000, "BOTELHO": 150000, "DORNELLES": 100000,
    "ESTEVES": 150000, "GOUVEIA": 100000, "LEITAO": 100000,
    "MEIRELES": 100000, "MEIRELLES": 100000, "MONTENEGRO": 100000,
    "NOBRE": 100000, "PEDROSO": 100000, "PENHA": 100000,
    "PROENCA": 100000, "QUARESMA": 80000, "RAPOSO": 80000,
    "SACRAMENTO": 100000, "SARAIVA": 100000, "SEABRA": 80000,
    "SERPA": 80000, "SIMOES": 150000, "TELES": 100000,
    "TRINDADE": 150000, "VELOSO": 100000, "VENTURA": 100000,
    "BRAZ": 100000, "NONATO": 80000, "FIRMINO": 100000, "GALDINO": 80000,
    "LEAO": 100000,
    # Tier 2 - common Brazilian
    "SILVA": 10000000, "SANTOS": 6000000, "PEREIRA": 3000000, "ALVES": 2500000,
    "RODRIGUES": 2000000, "OLIVEIRA": 4000000, "SOUZA": 3500000, "SOUSA": 1500000,
    "GOMES": 2000000, "RIBEIRO": 2000000, "SOARES": 1500000, "MARTINS": 1500000,
    "BARBOSA": 1200000, "VIEIRA": 1500000, "LOPES": 1200000, "FERNANDES": 1000000,
    "BATISTA": 800000, "DIAS": 1200000, "MOREIRA": 800000, "NUNES": 800000,
    "ALMEIDA": 1500000, "MENDES": 800000, "ARAUJO": 1200000, "CARDOSO": 600000,
    "MARQUES": 600000, "RAMOS": 800000, "MACHADO": 600000, "ROCHA": 800000,
    "SANTANA": 600000, "BORGES": 400000, "MONTEIRO": 400000, "CORREA": 500000,
    "CORREIA": 300000, "ANDRADE": 800000, "PINTO": 500000, "LEITE": 500000,
    "DUARTE": 400000, "FREITAS": 500000, "BARROS": 400000, "CAMPOS": 400000,
    "REIS": 400000, "MELO": 400000, "MELLO": 200000, "BRITO": 400000,
    "CARNEIRO": 250000, "SILVEIRA": 300000, "MEDEIROS": 300000, "FARIAS": 300000,
    "DANTAS": 200000, "BRAGA": 250000, "FONSECA": 300000, "AGUIAR": 250000,
    "AMARAL": 250000, "SAMPAIO": 200000, "PAIVA": 200000, "PORTO": 150000,
    "SALES": 200000, "PACHECO": 200000, "TOLEDO": 150000, "MAIA": 200000,
    "LEAL": 150000, "MACIEL": 200000, "MOTA": 200000, "BUENO": 200000,
    "DUTRA": 150000, "PAULA": 300000,
    "BANDEIRA": 150000, "BARBOZA": 200000, "BENTO": 200000,
    "CANDIDO": 200000, "COUTO": 150000, "CUSTODIO": 200000,
    "EVANGELISTA": 200000, "FRANCA": 200000, "INACIO": 200000,
    "MARIANO": 200000, "MATIAS": 200000, "MATOS": 200000,
    "MESSIAS": 200000, "MUNIZ": 200000, "NETO": 200000,
    "NOVAES": 150000, "PAULINO": 200000, "PONTES": 200000,
    "QUEIROGA": 150000, "RANGEL": 200000, "RESENDE": 200000,
    "SA": 200000, "SANTIAGO": 200000, "SENA": 200000,
    "SIMAO": 150000, "TELLES": 200000, "TEODORO": 200000,
    "VARGAS": 200000, "VAZ": 200000, "VEIGA": 200000,
    # Tier 3 - ambiguous (shared with Hispanic)
    "COSTA": 800000, "LIMA": 600000, "CRUZ": 400000,
    "GARCIA": 300000, "TORRES": 200000, "FRANCO": 200000,
    "ROSA": 300000, "CASTRO": 400000, "MIRANDA": 400000,
    "FELIX": 200000, "HENRIQUE": 200000, "APARECIDO": 200000,
}

# Brazilian-Americans (~1.9M) relative to Brazil's population: converts an
# in-Brazil bearer count into an estimated count of US-resident bearers, so
# the Brazilian mass is comparable with the US Census ethnicity masses.
BR_US_SCALE = 1.9 / 210

# Spain INE per-capita rate → estimated US-Hispanic bearers (62M US Hispanics
# vs 47M Spain residents). Rough, but puts the Hispanic first-name mass on the
# same US-bearer basis as SSA counts.
US_HISPANIC_SCALE = 62 / 47

# First names cannot use the US-bearer basis: SSA counts absorb US-born
# Brazilian children (most US THIAGOs *are* Brazilian-American), so treating
# SSA as "american" mass would erase distinctly Brazilian given names.
# Instead first names use per-capita distinctiveness (bearers per million of
# each source population), with pan-Latin staples discounted explicitly.
FIRSTNAME_SCALE = {"brazilian": 1 / 210, "american": 1 / 330, "hispanic": 1 / 47}

# Given names that are staples across the whole Spanish- AND Portuguese-
# speaking world. Per-capita frequency overstates them as Brazilian evidence
# (6% of Brazilian women are MARIA) — for US property records they identify
# "Latin" but not "Brazilian", so their Brazilian share is capped and the
# excess moved to Hispanic.
PAN_LATIN_STAPLE_MAX_BR = 0.45
PAN_LATIN_STAPLES = {
    "MARIA", "JOSE", "CARLOS", "LUIS", "ANTONIO", "MANUEL", "FRANCISCO",
    "FERNANDO", "PEDRO", "OSVALDO", "OSWALDO", "ROSA", "CARMEN", "ANA",
    "ANGELA", "TERESA", "MARTA", "SANDRA", "MONICA", "SILVIA", "GLORIA",
    "ALICIA", "ELENA", "RICARDO", "ROBERTO", "EDUARDO", "ALBERTO", "SERGIO",
    "MARIO", "JORGE", "RAUL", "HUGO", "OSCAR", "CESAR", "RAMON", "FELIX",
    "ALEX", "DANIEL", "DAVID", "GABRIEL", "RAFAEL", "SAMUEL", "ANDRES",
    "ORLANDO", "ROLANDO", "MARISA", "KARINE", "LILIAN", "SOLANGE",
    "MARCIA", "REINALDO", "GILBERTO", "STELLA",
}

# Pseudo-mass added to every class before normalizing, so a name seen in a
# single source can never emit a 1.0 probability. Surnames are in estimated
# US bearers; first names are in per-capita (bearers-per-million) units.
SURNAME_SMOOTHING = 25
FIRSTNAME_SMOOTHING = 2

# Caribbean/Puerto Rican given names that are absent from Spain INE (the only
# Hispanic first-name source) but common in Central Florida. Values are
# Spain-equivalent bearer counts so they flow through the same normalization.
CARIBBEAN_FIRST_NAME_SEEDS = {
    "MARITZA": 150000, "YESENIA": 150000, "XIOMARA": 120000, "MILAGROS": 150000,
    "LUZ": 150000, "MARIBEL": 150000, "MAYRA": 120000, "IVELISSE": 60000,
    "MIGDALIA": 60000, "NEREIDA": 60000, "AWILDA": 40000, "ODALYS": 40000,
    "YAJAIRA": 40000, "ZORAIDA": 40000, "LISSETTE": 60000, "NILDA": 60000,
    "IDALIA": 40000, "DAYANARA": 30000, "ELBA": 60000, "ZULMA": 40000,
    "YARITZA": 60000, "YANELYS": 30000, "YANELIS": 30000, "MIREIDYS": 20000,
    "YUDELKA": 30000, "YADIEL": 40000, "JADIEL": 20000, "YANDEL": 30000,
    "HERIBERTO": 100000, "EFRAIN": 100000, "ANIBAL": 60000, "MADELINE": 60000,
    "SELENIA": 20000, "MERIDA": 20000, "BALDOMERA": 15000, "YOLANDA": 150000,
}


def build_first_name_frequencies(first_names: pd.DataFrame, surnames: pd.DataFrame) -> dict:
    """Build per-name cross-country probability table for first names."""
    # Group by name and country, sum frequencies
    grouped = first_names.groupby(["name", "country"])["frequency"].sum().reset_index()

    # Inject Caribbean given-name seeds missing from Spain INE
    seed_rows = pd.DataFrame(
        [{"name": n, "country": "hispanic", "frequency": f}
         for n, f in CARIBBEAN_FIRST_NAME_SEEDS.items()]
    )
    grouped = pd.concat([grouped, seed_rows], ignore_index=True)
    grouped = grouped.groupby(["name", "country"])["frequency"].sum().reset_index()

    # Pre-compute total raw frequency per name (vectorized)
    total_by_name = grouped.groupby("name")["frequency"].sum()

    # Pivot to get name × country matrix
    pivot = grouped.pivot_table(index="name", columns="country", values="frequency", fill_value=0)

    # Per-capita distinctiveness basis (bearers per million of each source
    # population). Hispanic divides by Spain's population — the actual
    # coverage of the only Hispanic first-name source.
    for country in pivot.columns:
        pivot[country] = pivot[country] * FIRSTNAME_SCALE.get(country, 1.0)

    # Smooth: pseudo-mass per class so a single-source rare name (e.g. a
    # surname appearing 326 times as an IBGE given name) cannot reach 1.0
    pivot = pivot + FIRSTNAME_SMOOTHING

    # Compute probabilities (vectorized)
    row_sums = pivot.sum(axis=1)
    probs = pivot.div(row_sums, axis=0)

    # Filter by minimum occurrences and build output dict
    valid_names = total_by_name[total_by_name >= FREQUENCY_MIN_OCCURRENCES].index
    probs_filtered = probs.loc[probs.index.isin(valid_names)]

    # Suppress tokens that are predominantly surnames: if US Census surname
    # bearers dwarf given-name bearers, a first-name lookup is noise
    # (MATOS: 21k surname bearers vs 326 given-name bearers).
    us_surname_freq = (
        surnames[surnames["country"] == "american"]
        .groupby("name")["frequency"].sum()
    )
    surname_dominated = {
        name for name in probs_filtered.index
        if us_surname_freq.get(name, 0) > 3 * total_by_name.get(name, 1)
    }

    result = {}
    for name in probs_filtered.index:
        if name in surname_dominated:
            continue
        row = probs_filtered.loc[name]
        entry = {col: round(float(row[col]), PROBABILITY_DECIMALS)
                 for col in probs_filtered.columns if row[col] > 0.005}
        if not entry:
            continue
        br = entry.get("brazilian", 0)
        if name in PAN_LATIN_STAPLES and br > PAN_LATIN_STAPLE_MAX_BR:
            excess = br - PAN_LATIN_STAPLE_MAX_BR
            entry["brazilian"] = PAN_LATIN_STAPLE_MAX_BR
            entry["hispanic"] = round(entry.get("hispanic", 0) + excess, PROBABILITY_DECIMALS)
        result[name] = entry

    return result


def build_surname_frequencies(surnames: pd.DataFrame) -> dict:
    """Build per-surname cross-country probability table.

    All three classes are put on the same basis — estimated US-RESIDENT
    bearers — so the masses are comparable:
      - hispanic:  US Census bearer count x pcthispanic (direct measurement,
                   covers Puerto Rican/Caribbean/Mexican distributions)
      - american:  US Census bearer count x pctwhite, minus the estimated
                   Brazilian bearers (the census counts Brazilians as white)
      - brazilian: in-Brazil bearer estimate scaled by the Brazilian-American
                   population share (BR_US_SCALE)
    Spain INE contributes the Hispanic mass for surnames absent from the US
    Census (previously it was silently discarded whenever a US entry existed).
    """
    us_census = surnames[surnames["country"] == "american"].copy()
    spain = surnames[surnames["country"] == "hispanic"].copy()
    spain_freq_by_name = spain.groupby("name")["frequency"].sum()

    # Accumulate raw bearer masses per name, then normalize once
    masses: dict[str, dict[str, float]] = {}

    for _, row in us_census.iterrows():
        name = row["name"]
        freq = row["frequency"]
        if freq < FREQUENCY_MIN_OCCURRENCES:
            continue
        pcthisp = float(row.get("pcthispanic", 0) or 0)
        pctwhite = float(row.get("pctwhite", 0) or 0)

        br_bearers = BRAZILIAN_SURNAME_SEEDS.get(name, 0) * BR_US_SCALE
        hisp_bearers = freq * pcthisp / 100.0
        am_bearers = max(0.0, freq * pctwhite / 100.0 - br_bearers)

        masses[name] = {
            "brazilian": br_bearers,
            "hispanic": hisp_bearers,
            "american": am_bearers,
        }

    # Brazilian seeds absent from the US Census
    for name, freq in BRAZILIAN_SURNAME_SEEDS.items():
        if name not in masses:
            masses[name] = {
                "brazilian": freq * BR_US_SCALE,
                "hispanic": float(spain_freq_by_name.get(name, 0)),
                "american": 0.0,
            }

    # Spain INE surnames absent from the US Census (rare names): the Spain
    # bearer count stands in for the US-Hispanic bearer estimate
    for name, freq in spain_freq_by_name.items():
        if name not in masses:
            masses[name] = {"brazilian": 0.0, "hispanic": float(freq), "american": 0.0}

    result = {}
    for name, raw in masses.items():
        smoothed = {k: v + SURNAME_SMOOTHING for k, v in raw.items()}
        total = sum(smoothed.values())
        entry = {k: round(v / total, PROBABILITY_DECIMALS)
                 for k, v in smoothed.items() if v / total > 0.005}
        if entry:
            result[name] = entry

    return result


def merge_all():
    """Build and save frequency tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading normalized data...")
    first_names = pd.read_csv(PROCESSED_DIR / "all_first_names.csv")
    surnames = pd.read_csv(PROCESSED_DIR / "all_surnames.csv")

    print(f"  First names: {len(first_names)} rows")
    print(f"  Surnames: {len(surnames)} rows")

    print("\nBuilding first name frequency table...")
    fn_probs = build_first_name_frequencies(first_names, surnames)
    print(f"  {len(fn_probs)} names in frequency table")

    print("\nBuilding surname frequency table...")
    sn_probs = build_surname_frequencies(surnames)
    print(f"  {len(sn_probs)} surnames in frequency table")

    # Save
    output = {
        "version": MODEL_VERSION,
        "first_names": fn_probs,
        "surnames": sn_probs,
    }

    output_path = OUTPUT_DIR / "frequency_tables.json"
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")

    # Validation
    print("\n=== Validation ===")
    for name in ["JOAO", "MARIA", "JOHN", "GARCIA", "FERREIRA", "SILVA", "SMITH", "GUADALUPE"]:
        fn_entry = fn_probs.get(name, {})
        sn_entry = sn_probs.get(name, {})
        if fn_entry:
            print(f"  {name} (first): {fn_entry}")
        if sn_entry:
            print(f"  {name} (surname): {sn_entry}")
        if not fn_entry and not sn_entry:
            print(f"  {name}: not found")

    return fn_probs, sn_probs


def load_eval_full_names() -> set[str]:
    """Read normalized eval full names if Builder A3 has created them."""
    path = PROJECT_ROOT / "data" / "eval" / "eval_names.csv"
    if not path.exists():
        return set()

    eval_names: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return set()
        preferred = [
            "name_normalized", "full_name", "name", "owner_name", "owner",
            "person_name", "display_name", "label",
        ]
        cols = [c for c in preferred if c in reader.fieldnames]
        if not cols:
            cols = [reader.fieldnames[0]]
        for row in reader:
            parts = [str(row.get(col, "")) for col in cols if row.get(col)]
            normalized = clean_name(" ".join(parts))
            if normalized:
                eval_names.add(normalized)
    return eval_names


def add_count(masses: dict, name: str, class_name: str, count: float) -> None:
    if name and class_name in COUNTRY_TABLE_CLASSES and count > 0:
        masses[name][class_name] += float(count)


def add_existing_br_sources(given_masses: dict, surname_masses: dict) -> None:
    first_path = PROCESSED_DIR / "all_first_names.csv"
    if first_path.exists():
        first = pd.read_csv(first_path)
        br_first = first[first["country"] == "brazilian"]
        for _, row in br_first.iterrows():
            add_count(given_masses, clean_name(str(row["name"])), "br", float(row["frequency"]))

    for name, freq in BRAZILIAN_SURNAME_SEEDS.items():
        add_count(surname_masses, clean_name(name), "br", float(freq))


def add_hispanic_counterweight_sources(given_masses: dict, surname_masses: dict) -> None:
    """Inject a non-exported Hispanic pseudo-class for country attribution.

    The Stage-2 country table is exported over Lusophone countries, but the
    normalization must see Spanish/US-Hispanic mass or shared surnames such as
    DELGADO, DIAZ, FREIRE, VEIGA, and MONTEIRO become falsely confident CV/BR
    tokens. This pseudo-class is consumed only as counter-evidence.
    """
    first_path = PROCESSED_DIR / "all_first_names.csv"
    if first_path.exists():
        first = pd.read_csv(first_path)
        hispanic_first = first[first["country"] == "hispanic"]
        for _, row in hispanic_first.iterrows():
            add_count(given_masses, clean_name(str(row["name"])), "hispanic", float(row["frequency"]))

    surname_path = PROCESSED_DIR / "all_surnames.csv"
    if not surname_path.exists():
        return
    surnames = pd.read_csv(surname_path)
    spain = surnames[surnames["country"] == "hispanic"]
    for _, row in spain.iterrows():
        add_count(surname_masses, clean_name(str(row["name"])), "hispanic", float(row["frequency"]))
    us = surnames[surnames["country"] == "american"]
    for _, row in us.iterrows():
        pcthisp = float(row.get("pcthispanic", 0) or 0)
        if pcthisp <= 0:
            continue
        add_count(
            surname_masses,
            clean_name(str(row["name"])),
            "hispanic",
            float(row["frequency"]) * pcthisp / 100.0,
        )


def add_lusophone_sources(given_masses: dict, surname_masses: dict) -> dict:
    given_path = PROCESSED_DIR / "lusophone_given_names.csv"
    surname_path = PROCESSED_DIR / "lusophone_surnames.csv"
    people_path = PROCESSED_DIR / "lusophone_wikidata_people.csv"

    if not given_path.exists() or not surname_path.exists() or not people_path.exists():
        normalize_lusophone()

    eval_full_names = load_eval_full_names()
    overlap_names: set[str] = set()
    dropped_token_rows = 0

    for path, target in [(given_path, given_masses), (surname_path, surname_masses)]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            full_name = clean_name(str(row.get("full_name", "")))
            if full_name and full_name in eval_full_names:
                overlap_names.add(full_name)
                dropped_token_rows += 1
                continue
            add_count(
                target,
                clean_name(str(row.get("name", ""))),
                str(row.get("country_class", "")),
                float(row.get("frequency", 0) or 0),
            )

    train_full_names: set[str] = set()
    if people_path.exists():
        people = pd.read_csv(people_path)
        train = people[people["split"] == "train"]
        for _, row in train.iterrows():
            full_name = clean_name(str(row.get("full_name", "")))
            if full_name:
                train_full_names.add(full_name)

    original_overlap = train_full_names & eval_full_names
    filtered_train_full_names = train_full_names - eval_full_names
    return {
        "eval_names_file_present": bool(eval_full_names),
        "overlap_before_drop": len(original_overlap),
        "overlap_after_drop": len(filtered_train_full_names & eval_full_names),
        "dropped_full_names": len(original_overlap),
        "dropped_token_rows": dropped_token_rows,
        "overlap_sample": sorted(original_overlap)[:20],
    }


def drop_surname_dominated_given_rows(given_masses: dict, surname_masses: dict) -> int:
    dropped = 0
    for name in list(given_masses):
        given_total = sum(given_masses[name].values())
        surname_total = sum(surname_masses.get(name, {}).values())
        if surname_total > 3 * max(given_total, 1.0):
            del given_masses[name]
            dropped += 1
    return dropped


def class_totals(masses: dict) -> dict[str, float]:
    totals = {cls: 0.0 for cls in COUNTRY_TABLE_CLASSES}
    for by_class in masses.values():
        for cls, count in by_class.items():
            totals[cls] += count
    return totals


def normalized_observed_weights(by_class: dict[str, float], totals: dict[str, float]) -> dict[str, float]:
    shares = {
        cls: (count / totals[cls])
        for cls, count in by_class.items()
        if count > 0 and totals.get(cls, 0) > 0
    }
    total_share = sum(shares.values())
    if total_share <= 0:
        return {}
    return {
        cls: round(share / total_share, PROBABILITY_DECIMALS)
        for cls, share in sorted(shares.items())
        if share / total_share > 0.001
    }


def absolute_observed_weights(by_class: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in by_class.values() if value > 0)
    if total <= 0:
        return {}
    return {
        cls: round(value / total, PROBABILITY_DECIMALS)
        for cls, value in sorted(by_class.items())
        if value > 0 and value / total > 0.001
    }


def collect_boosts(table_type: str) -> dict[str, dict[str, dict]]:
    boosts_by_name: dict[str, dict[str, dict]] = defaultdict(dict)
    if table_type == "given_names":
        boost_groups = ["given_names", "compound_given_names"]
    else:
        boost_groups = ["surnames"]

    for group in boost_groups:
        for cls, spec in CURATED_LUSOPHONE_BOOSTS.get(group, {}).items():
            for token in spec["tokens"]:
                name = clean_name(token)
                boosts_by_name[name][cls] = {
                    "boost": float(spec["boost"]),
                    "source": f"curated_{group}",
                    "confidence": spec["confidence"],
                }
    return boosts_by_name


def boost_weights(boost_entry: dict[str, dict]) -> dict[str, float]:
    total = sum(float(v["boost"]) for v in boost_entry.values())
    if total <= 0:
        return {}
    return {
        cls: round(float(v["boost"]) / total, PROBABILITY_DECIMALS)
        for cls, v in sorted(boost_entry.items())
    }


def protect_country_specific_weight(name: str, weights: dict[str, float]) -> dict[str, float]:
    home_class = PROTECTED_COUNTRY_SPECIFIC_SURNAMES.get(name)
    if not home_class:
        return weights
    if weights.get(home_class, 0.0) >= PROTECTED_HOME_MIN_WEIGHT:
        return weights

    protected = dict(weights)
    others_total = sum(value for cls, value in protected.items() if cls != home_class and value > 0)
    protected[home_class] = PROTECTED_HOME_MIN_WEIGHT
    remaining = 1.0 - PROTECTED_HOME_MIN_WEIGHT
    if others_total <= 0:
        return {home_class: 1.0}
    for cls in list(protected):
        if cls == home_class:
            continue
        protected[cls] = protected[cls] / others_total * remaining
    return protected


def combined_weights(
    observed: dict[str, float],
    boosts: dict[str, dict],
    name: str = "",
    table_type: str = "",
) -> dict[str, float]:
    combined = defaultdict(float)
    for cls, weight in observed.items():
        combined[cls] += weight
    # Boosts are bounded pseudo-evidence, not raw counts. A single-class boost
    # can move an otherwise unseen token but observed data still participates.
    hispanic_weight = observed.get("hispanic", 0.0)
    pan_lusophone_weight = sum(observed.get(cls, 0.0) for cls in ("br", "pt", "palop_other"))
    for cls, spec in boosts.items():
        boost = min(float(spec["boost"]), 1.5)
        # Curated CV/AO/MZ seeds are hints, not a license to override clear
        # Hispanic or pan-Lusophone base rates in real property data.
        if cls in {"cv", "ao", "mz"} and (hispanic_weight >= 0.35 or pan_lusophone_weight >= 0.60):
            boost *= 0.15
        combined[cls] += boost
    if table_type == "surnames":
        combined = defaultdict(float, protect_country_specific_weight(name, dict(combined)))
    total = sum(combined.values())
    if total <= 0:
        return {}
    return {
        cls: round(value / total, PROBABILITY_DECIMALS)
        for cls, value in sorted(combined.items())
        if value / total > 0.001
    }


def build_country_table_entries(masses: dict, table_type: str) -> tuple[dict, dict]:
    totals = class_totals(masses)
    boosts_by_name = collect_boosts(table_type)
    names = set(masses) | set(boosts_by_name)
    entries = {}
    effective_vocab = {cls: 0 for cls in LUSOPHONE_CLASSES}

    for name in sorted(names):
        by_class = dict(sorted(masses.get(name, {}).items()))
        if table_type == "surnames":
            observed = absolute_observed_weights(by_class)
        else:
            observed = normalized_observed_weights(by_class, totals)
        boosts = boosts_by_name.get(name, {})
        combined = combined_weights(observed, boosts, name=name, table_type=table_type)
        if not combined:
            continue
        for cls, weight in combined.items():
            if cls in LUSOPHONE_CLASSES and weight > 0:
                effective_vocab[cls] += 1
        entry = {
            "counts": {cls: int(count) if count.is_integer() else round(count, 3)
                       for cls, count in by_class.items() if count > 0},
            "weights": observed,
            "boosts": boosts,
            "combined_weights": combined,
        }
        entries[name] = entry

    return entries, {
        "class_totals": {cls: round(totals[cls], 3) for cls in LUSOPHONE_CLASSES},
        "effective_vocab": effective_vocab,
    }


def merge_lusophone_country_tables() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    given_masses: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    surname_masses: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    print("Building Lusophone country attribution tables...")
    add_existing_br_sources(given_masses, surname_masses)
    overlap_report = add_lusophone_sources(given_masses, surname_masses)
    add_hispanic_counterweight_sources(given_masses, surname_masses)
    dropped_given = drop_surname_dominated_given_rows(given_masses, surname_masses)

    given_entries, given_stats = build_country_table_entries(given_masses, "given_names")
    surname_entries, surname_stats = build_country_table_entries(surname_masses, "surnames")

    output = {
        "version": MODEL_VERSION,
        "classes": list(LUSOPHONE_CLASSES),
        "unattributable_prior": LUSOPHONE_UNATTRIBUTABLE_PRIOR,
        "weighting_scheme": (
            "Surnames use observed absolute-mass weights with a Hispanic pseudo-class "
            "counterweight; protected high-specificity CV/MZ/AO surname tokens cap "
            "base-rate dilution so their home class remains country-specific. Given "
            "names use within-class normalization. Curated boosts are tagged "
            "pseudo-evidence kept separate from counts and exposed in combined_weights."
        ),
        "sources": {
            "br_given": "existing Brasil.IO/IBGE normalized first-name table",
            "br_surname": "existing BRAZILIAN_SURNAME_SEEDS from merge.py",
            "pt_given": "IRN top-20 annual CSVs plus admissible-name whitelist",
            "wikidata": "CC0 human citizenship corpus, deterministic QID train/eval split",
            "curated_boosts": "constants.py reviewable lists from A2 research; no Forebears scraping",
            "hispanic_pseudo_class": "Spain INE surnames/given names plus US Census pcthispanic counterweight; not exported as an origin class",
            "protected_specific_surnames": "C4 table revision: SEMEDO/FORTES/VARELA/DE PINA/DA LUZ/EVORA/FURTADO plus curated MZ/AO clusters keep home-class specificity; DELGADO/MORENO/SANCHES excluded",
        },
        "eval_split": overlap_report,
        "data_cleaning": {
            "surname_dominated_given_rows_dropped": dropped_given,
        },
        "stats": {
            "given_names": given_stats,
            "surnames": surname_stats,
        },
        "given_names": given_entries,
        "surnames": surname_entries,
    }

    output_path = OUTPUT_DIR / "country_tables.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"), ensure_ascii=True)

    print(f"  given names: {len(given_entries)}")
    print(f"  surnames: {len(surname_entries)}")
    print(f"  eval overlap before drop: {overlap_report['overlap_before_drop']}")
    print(f"  saved: {output_path} ({output_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["base", "lusophone", "all"], default="all")
    args = parser.parse_args()
    if args.only == "base":
        merge_all()
    elif args.only == "lusophone":
        merge_lusophone_country_tables()
    else:
        merge_all()
        merge_lusophone_country_tables()
