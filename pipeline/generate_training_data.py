"""Generate synthetic full names for training the classifier."""

import json
import random

import pandas as pd
import numpy as np

from config import PROCESSED_DIR, OUTPUT_DIR, TRAINING_SAMPLES_PER_CLASS, RANDOM_SEED

# Portuguese prepositions and suffixes for Brazilian name generation
PT_PREPOSITIONS = ["DA", "DO", "DOS", "DAS", "DE"]
BR_SUFFIXES = ["JUNIOR", "NETO", "FILHO"]
BR_COMPOUND_FIRST = [
    "ANA PAULA", "ANA CAROLINA", "ANA CLAUDIA", "ANA LUCIA", "ANA MARIA",
    "MARIA FERNANDA", "MARIA EDUARDA", "MARIA LUIZA", "MARIA CLARA",
    "JOSE CARLOS", "JOSE ROBERTO", "LUIZ FERNANDO", "LUIZ CARLOS",
    "JOAO PEDRO", "JOAO VICTOR", "JOAO PAULO", "JOAO GABRIEL",
    "PEDRO HENRIQUE", "MARCOS ANTONIO", "CARLOS EDUARDO", "PAULO ROBERTO",
]

# Spanish preposition patterns
ES_PREPOSITIONS = ["DE LA", "DE LOS", "DE LAS", "DEL"]

# Merged preposition forms (Florida property record artifacts)
MERGED_PREPS = {
    "DASILVA": "DA SILVA", "DESOUZA": "DE SOUZA", "DESOUSA": "DE SOUSA",
    "DEOLIVEIRA": "DE OLIVEIRA", "DEALMEIDA": "DE ALMEIDA",
    "DOSANTOS": "DOS SANTOS", "DOSSANTOS": "DOS SANTOS",
    "DEPAULA": "DE PAULA", "DEFREITAS": "DE FREITAS",
    "DELIMA": "DE LIMA", "DEMOURA": "DE MOURA", "DEBRITO": "DE BRITO",
}

BR_HARD_SURNAMES = [
    "SILVA", "SANTOS", "OLIVEIRA", "SOUZA", "PEREIRA", "FERREIRA",
    "COSTA", "LIMA", "MARTINS", "RODRIGUES", "ALMEIDA", "RIBEIRO",
    "CARVALHO", "GOMES", "NASCIMENTO", "TEIXEIRA", "MAIA", "MENDES",
]

ANGLO_FIRST_NAMES = [
    "JOHN", "MICHAEL", "ROBERT", "DAVID", "JAMES", "WILLIAM", "RICHARD",
    "THOMAS", "CHARLES", "DANIEL", "MATTHEW", "MARK", "PAUL", "STEVEN",
    "MARY", "PATRICIA", "JENNIFER", "LINDA", "ELIZABETH", "BARBARA",
    "SUSAN", "KAREN", "NANCY", "LISA", "SARAH", "LAURA",
]

HISPANIC_HARD_SURNAMES = [
    "GARCIA", "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ",
    "PEREZ", "SANCHEZ", "RAMIREZ", "TORRES", "RIVERA", "MORALES",
    "CRUZ", "ORTIZ", "FLORES", "GOMEZ", "MENDOZA", "CASTILLO",
    "VAZQUEZ", "VELAZQUEZ", "JIMENEZ", "FERNANDEZ", "DOMINGUEZ",
    "GUTIERREZ", "ALVAREZ", "MENDEZ", "CHAVEZ", "NEGRON", "FIGUEROA",
    "ROSADO", "COLON", "MALDONADO", "BURGOS", "FELICIANO", "QUINONES",
    "ZAYAS", "ESQUIVEL", "RIOS",
]

HISPANIC_HARD_FIRST_NAMES = [
    "MARIA", "JOSE", "JUAN", "CARLOS", "LUIS", "MIGUEL", "JESUS",
    "GUADALUPE", "CARMEN", "ROSA", "MARGARITA", "FRANCISCO", "JORGE",
    "ALEJANDRO", "MANUEL", "RAFAEL", "ANGEL", "PEDRO",
    "JAVIER", "YOLANDA", "XIOMARA", "YARITZA", "YESENIA", "MARITZA",
    "IVELISSE", "MIGDALIA", "HERIBERTO", "EFRAIN", "MILAGROS", "IRMA",
    "YANELYS", "MIREIDYS", "LUZ", "MAYRA", "NEREIDA",
]

# Surnames shared between Portuguese and Spanish that are common among
# Puerto Ricans/Caribbeans — the given name decides the class. These drove
# the 2026-07 audit's Very-High false positives.
SHARED_IBERIAN_SURNAMES = [
    "SANTOS", "MATOS", "BATISTA", "PAULINO", "PACHECO", "FONSECA",
    "BRITO", "SANTANA", "ROSA", "RAMOS", "CASTRO", "DUARTE",
    "SANTIAGO", "MUNIZ", "FELIX", "SILVA", "ROCHA", "MIRANDA",
]

BR_HARD_FIRST_NAMES = [
    "THIAGO", "JOAO", "GUSTAVO", "LUCIANO", "LEANDRO", "MARCELO",
    "VINICIUS", "FABIO", "JULIANA", "FERNANDA", "LUCIANA", "ROSANGELA",
    "ANA PAULA", "JOAO PEDRO", "MARIA FERNANDA", "CLEUDIMAR", "GENIVALDO",
    "GIVANILDO", "EDILEUSA", "WANDERLEI", "ROSILENE",
]


def generate_hard_cases(rng_np, n_each=12_000) -> list[dict]:
    """Generate edge cases that real Florida property records often confuse."""
    records: list[dict] = []

    for _ in range(n_each):
        surname = rng_np.choice(BR_HARD_SURNAMES)
        first = rng_np.choice(ANGLO_FIRST_NAMES)
        middle = rng_np.choice(ANGLO_FIRST_NAMES) if rng_np.random() < 0.20 else ""
        name = " ".join(p for p in [surname, first, middle] if p)
        records.append({"name": name, "label": "american", "source": "hard_negative_br_surname_anglo_first"})

    spanish_patterns = [
        lambda sn, fn: f"DE LOS {sn} {fn}",
        lambda sn, fn: f"DE LA {sn} {fn}",
        lambda sn, fn: f"{sn} DE LA CRUZ {fn}",
        lambda sn, fn: f"{sn} DE LOS SANTOS {fn}",
        lambda sn, fn: f"{sn} {fn}",
    ]
    for _ in range(n_each):
        surname = rng_np.choice(HISPANIC_HARD_SURNAMES)
        first = rng_np.choice(HISPANIC_HARD_FIRST_NAMES)
        name = rng_np.choice(spanish_patterns)(surname, first)
        records.append({"name": name, "label": "hispanic", "source": "hard_negative_hispanic"})

    # Shared Iberian surname + distinctly Hispanic given name → hispanic.
    # Teaches the model that SANTOS/MATOS/BATISTA etc. are NOT Brazilian
    # evidence unless a Brazilian given name corroborates.
    for _ in range(n_each):
        surname = rng_np.choice(SHARED_IBERIAN_SURNAMES)
        first = rng_np.choice(HISPANIC_HARD_FIRST_NAMES)
        r = rng_np.random()
        if r < 0.30:
            second = rng_np.choice(HISPANIC_HARD_SURNAMES)
            name = f"{surname} {second} {first}"
        elif r < 0.45:
            second = rng_np.choice(SHARED_IBERIAN_SURNAMES)
            name = f"{surname} {second} {first}"
        else:
            name = f"{surname} {first}"
        records.append({"name": name, "label": "hispanic", "source": "hard_negative_shared_surname_hispanic_first"})

    # Shared Iberian surname + distinctly Brazilian given name → brazilian
    for _ in range(n_each // 2):
        surname = rng_np.choice(SHARED_IBERIAN_SURNAMES)
        first = rng_np.choice(BR_HARD_FIRST_NAMES)
        name = f"{surname} {first}"
        records.append({"name": name, "label": "brazilian", "source": "hard_positive_shared_surname_br_first"})

    br_firsts = BR_HARD_FIRST_NAMES
    br_preps = ["DA", "DE", "DO", "DOS", "DAS"]
    for _ in range(n_each // 2):
        surname = rng_np.choice(BR_HARD_SURNAMES)
        second = rng_np.choice(BR_HARD_SURNAMES)
        first = rng_np.choice(br_firsts)
        prep = rng_np.choice(br_preps)
        if rng_np.random() < 0.5:
            name = f"{prep} {surname} {first}"
        elif rng_np.random() < 0.75:
            name = f"{surname} {first} {prep} {second}"
        else:
            name = f"{first} {surname}"
        records.append({"name": name, "label": "brazilian", "source": "hard_positive_brazilian"})

    return records


def load_name_lists() -> dict:
    """Load frequency-weighted name lists per country."""
    first_names = pd.read_csv(PROCESSED_DIR / "all_first_names.csv")
    surnames = pd.read_csv(PROCESSED_DIR / "all_surnames.csv")

    result = {}
    for country in ["brazilian", "american", "hispanic"]:
        fn = first_names[first_names["country"] == country].copy()
        fn = fn[fn["frequency"] > 0].dropna(subset=["name", "frequency"])
        fn["weight"] = fn["frequency"] / fn["frequency"].sum()

        if country == "brazilian":
            # For Brazilian surnames, use the seed data from merge.py
            with open(OUTPUT_DIR / "frequency_tables.json") as f:
                freq_tables = json.load(f)
            br_surnames = []
            for name, probs in freq_tables["surnames"].items():
                br_prob = probs.get("brazilian", 0)
                if br_prob > 0.3:  # Strongly Brazilian
                    br_surnames.append({"name": name, "frequency": int(br_prob * 1000000)})
            sn = pd.DataFrame(br_surnames)
        elif country == "hispanic":
            sn = surnames[surnames["country"] == "hispanic"].copy()
            sn = sn[sn["frequency"] > 0].dropna(subset=["name", "frequency"])
            # Also add US Census surnames with high Hispanic percentage
            us_hisp = surnames[
                (surnames["country"] == "american") &
                (surnames.get("pcthispanic", pd.Series(dtype=float)).fillna(0).astype(float) > 50)
            ].copy()
            if not us_hisp.empty:
                us_hisp["country"] = "hispanic"
                sn = pd.concat([sn, us_hisp[["name", "country", "frequency"]]], ignore_index=True)
        else:
            sn = surnames[surnames["country"] == "american"].copy()
            sn = sn[sn["frequency"] > 0].dropna(subset=["name", "frequency"])
            # Filter to mostly non-Hispanic American surnames
            if "pcthispanic" in sn.columns:
                sn_pcthisp = sn["pcthispanic"].fillna(0)
                sn_pcthisp = pd.to_numeric(sn_pcthisp, errors="coerce").fillna(0)
                sn = sn[sn_pcthisp < 30]

        if not sn.empty:
            sn = sn.groupby("name", as_index=False)["frequency"].sum()
            sn["weight"] = sn["frequency"] / sn["frequency"].sum()

        result[country] = {
            "first_names": fn[["name", "weight"]].values,
            "surnames": sn[["name", "weight"]].values if not sn.empty else np.array([["UNKNOWN", 1.0]]),
        }

    return result


def batch_generate_names(rng_np, label, fn_names, fn_weights, sn_names, sn_weights, n):
    """Generate n names for a given label using vectorized numpy sampling."""
    # Pre-sample all random choices at once (much faster than per-name)
    fn_indices = rng_np.choice(len(fn_names), size=n, p=fn_weights)
    sn_indices = rng_np.choice(len(sn_names), size=n, p=sn_weights)
    randoms = rng_np.random(size=(n, 5))  # 5 random values per name

    names = []
    merged_keys = list(MERGED_PREPS.keys())

    for i in range(n):
        first = fn_names[fn_indices[i]]
        surname = sn_names[sn_indices[i]]

        if label == "brazilian":
            # Compound first name override
            if randoms[i, 0] < 0.08:
                first = BR_COMPOUND_FIRST[int(randoms[i, 1] * len(BR_COMPOUND_FIRST)) % len(BR_COMPOUND_FIRST)]

            # Merged preposition form
            if randoms[i, 4] < 0.08:
                merged = merged_keys[int(randoms[i, 1] * len(merged_keys)) % len(merged_keys)]
                names.append(f"{merged} {first}")
                continue

            parts = []
            # Second surname
            if randoms[i, 1] < 0.15:
                sn2_idx = rng_np.integers(len(sn_names))
                parts.append(sn_names[sn2_idx])

            # Preposition
            if randoms[i, 2] < 0.25:
                parts.append(PT_PREPOSITIONS[int(randoms[i, 3] * len(PT_PREPOSITIONS)) % len(PT_PREPOSITIONS)])

            parts.append(surname)
            parts.append(first)

            # Suffix
            if randoms[i, 3] < 0.04:
                parts.append(BR_SUFFIXES[int(randoms[i, 4] * len(BR_SUFFIXES)) % len(BR_SUFFIXES)])

            names.append(" ".join(parts))

        elif label == "hispanic":
            parts = []
            # Spanish preposition (rare)
            if randoms[i, 0] < 0.03:
                parts.append(ES_PREPOSITIONS[int(randoms[i, 1] * len(ES_PREPOSITIONS)) % len(ES_PREPOSITIONS)])
            parts.append(surname)
            # Double surname
            if randoms[i, 2] < 0.20:
                sn2_idx = rng_np.integers(len(sn_names))
                parts.append(sn_names[sn2_idx])
            parts.append(first)
            names.append(" ".join(parts))

        else:  # american
            if randoms[i, 0] < 0.10:
                middle = chr(65 + int(randoms[i, 1] * 26) % 26)
                names.append(f"{surname} {first} {middle}")
            else:
                names.append(f"{surname} {first}")

    return names


def _lusophone_country_training_rows(rng_np, n_per_class=80_000) -> list[dict]:
    """Sample B's train-pool corpora evenly across Lusophone countries.

    The final Stage-1 label is the gate label (lusophone), but `source`
    keeps the country class so the sampling decision remains auditable.
    """
    path = PROCESSED_DIR / "lusophone_wikidata_people.csv"
    if not path.exists():
        return []

    people = pd.read_csv(path)
    people = people[people["split"] == "train"].dropna(subset=["full_name", "country_class"])
    people["full_name"] = people["full_name"].astype(str).str.strip()
    people["given_name"] = people["given_name"].fillna("").astype(str).str.strip()
    people["family_name"] = people["family_name"].fillna("").astype(str).str.strip()

    rows: list[dict] = []
    classes = ["br", "pt", "cv", "ao", "mz", "palop_other"]
    for country_class in classes:
        pool = people[people["country_class"] == country_class]
        variants: list[str] = []
        for _, row in pool.iterrows():
            full_name = " ".join(str(row["full_name"]).split())
            if full_name:
                variants.append(full_name)
            given = " ".join(str(row["given_name"]).split())
            family = " ".join(str(row["family_name"]).split())
            if given and family:
                variants.append(f"{family} {given}")
        variants = sorted(set(variants))
        if not variants:
            continue
        indices = rng_np.choice(len(variants), size=n_per_class, replace=True)
        for idx in indices:
            rows.append(
                {
                    "name": variants[int(idx)],
                    "label": "lusophone",
                    "source": f"lusophone_train_pool_{country_class}",
                }
            )
    return rows


def generate_training_data():
    """Generate all synthetic training data."""
    rng_np = np.random.default_rng(RANDOM_SEED)

    print("Loading name lists...")
    name_lists = load_name_lists()

    for country, data in name_lists.items():
        print(f"  {country}: {len(data['first_names'])} first names, {len(data['surnames'])} surnames")

    n = TRAINING_SAMPLES_PER_CLASS
    all_records = []

    for label in ["brazilian", "hispanic", "american"]:
        print(f"\nGenerating {n:,} {label} names...")
        data = name_lists[label]
        fn_names = data["first_names"][:, 0]
        fn_weights = data["first_names"][:, 1].astype(float)
        fn_weights = fn_weights / fn_weights.sum()  # Ensure normalized

        sn_names = data["surnames"][:, 0]
        sn_weights = data["surnames"][:, 1].astype(float)
        sn_weights = sn_weights / sn_weights.sum()

        names = batch_generate_names(rng_np, label, fn_names, fn_weights, sn_names, sn_weights, n)
        for name in names:
            output_label = "lusophone" if label == "brazilian" else label
            source = "synthetic_brazilian_as_lusophone" if label == "brazilian" else "synthetic"
            all_records.append({"name": name, "label": output_label, "source": source})
        print(f"  Done: {len(names):,} names")

    hard_cases = generate_hard_cases(rng_np)
    for record in hard_cases:
        if record["label"] == "brazilian":
            record["label"] = "lusophone"
            record["source"] = record["source"].replace("brazilian", "lusophone").replace("br_", "lusophone_")
    all_records.extend(hard_cases)
    print(f"\nAdded hard cases: {len(hard_cases):,}")

    lusophone_rows = _lusophone_country_training_rows(rng_np)
    all_records.extend(lusophone_rows)
    print(f"Added Lusophone country train-pool rows: {len(lusophone_rows):,}")

    # Shuffle
    rng_np.shuffle(all_records)
    df = pd.DataFrame(all_records)

    # Split: 80/10/10
    n_total = len(df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"\n=== Training Data Summary ===")
    print(f"  Total: {n_total:,} names")
    print(f"  Train: {len(train):,}")
    print(f"  Val:   {len(val):,}")
    print(f"  Test:  {len(test):,}")
    print(f"\n  Class distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label}: {count:,}")

    # Sample inspection
    print(f"\n  Sample names:")
    for label in ["lusophone", "hispanic", "american"]:
        samples = df[df["label"] == label].head(5)["name"].tolist()
        print(f"    {label}:")
        for s in samples:
            print(f"      {s}")

    return df


if __name__ == "__main__":
    generate_training_data()
