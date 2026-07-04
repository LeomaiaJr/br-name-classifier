"""Normalize all census datasets into a unified format."""

import re
import hashlib
import unicodedata
from pathlib import Path

import pandas as pd

from config import RAW_DIR, LUSOPHONE_RAW_DIR, PROCESSED_DIR

NAME_PARTICLES = {
    "DA", "DAS", "DE", "DO", "DOS", "E", "DEL", "DELA", "LA", "LAS", "LOS",
}


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_name(name: str) -> str:
    cleaned = strip_accents(name).upper().replace(",", " ")
    return re.sub(r"\s+", " ", cleaned.strip())


def clean_optional_name(value) -> str:
    if pd.isna(value):
        return ""
    return clean_name(str(value))


def split_person_pool(qid: str) -> str:
    """Deterministic person-level split: 50% train-pool, 50% eval-pool."""
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return "train" if int(digest[:8], 16) % 2 == 0 else "eval"


def fallback_given_family(full_name: str, given: str, family: str) -> tuple[str, str]:
    """Use Wikidata properties when present, otherwise derive edge tokens."""
    if given and family:
        return given, family
    tokens = [t for t in clean_name(full_name).split() if t and not t.startswith("Q")]
    if not tokens:
        return given, family
    if not given:
        given = tokens[0]
    if not family:
        for token in reversed(tokens[1:] or tokens):
            if token not in NAME_PARTICLES:
                family = token
                break
    return given, family


def normalize_brasil_io() -> pd.DataFrame:
    path = RAW_DIR / "brasil_io_nomes.csv"
    if not path.exists():
        print("  [SKIP] brasil_io_nomes.csv not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        name = clean_name(str(row.get("first_name", "")))
        freq = int(row.get("frequency_total", 0) or 0)
        if name and freq > 0 and len(name) >= 2:
            records.append({"name": name, "country": "brazilian", "frequency": freq,
                            "gender": str(row.get("classification", "")), "type": "first_name"})
    result = pd.DataFrame(records)
    result = result.groupby(["name", "country", "type"], as_index=False).agg({"frequency": "sum", "gender": "first"})
    print(f"  Brasil.IO: {len(result)} unique first names")
    return result


def normalize_ssa() -> pd.DataFrame:
    ssa_dir = RAW_DIR / "ssa_names"
    if not ssa_dir.exists():
        print("  [SKIP] ssa_names/ not found")
        return pd.DataFrame()
    name_freq: dict[str, int] = {}
    for txt_file in sorted(ssa_dir.glob("yob*.txt")):
        with open(txt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    cleaned = clean_name(parts[0])
                    if cleaned and len(cleaned) >= 2:
                        name_freq[cleaned] = name_freq.get(cleaned, 0) + int(parts[2])
    records = [{"name": n, "country": "american", "frequency": f, "gender": "", "type": "first_name"}
               for n, f in name_freq.items()]
    print(f"  US SSA: {len(records)} unique first names")
    return pd.DataFrame(records)


def normalize_us_census_surnames() -> pd.DataFrame:
    path = RAW_DIR / "us_census_surnames.csv"
    if not path.exists():
        print("  [SKIP] us_census_surnames.csv not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        name = clean_name(str(row.get("name", "")))
        try:
            count = int(float(row.get("count", 0)))
        except (ValueError, TypeError):
            continue
        if not name or count == 0 or len(name) < 2:
            continue
        pcthispanic = pctwhite = 0.0
        for col, target in [("pcthispanic", "pcthispanic"), ("pctwhite", "pctwhite")]:
            val = row.get(col, 0)
            try:
                if pd.notna(val) and val != "(S)":
                    if target == "pcthispanic":
                        pcthispanic = float(val)
                    else:
                        pctwhite = float(val)
            except (ValueError, TypeError):
                pass
        records.append({"name": name, "country": "american", "frequency": count,
                        "gender": "", "type": "surname", "pcthispanic": pcthispanic, "pctwhite": pctwhite})
    result = pd.DataFrame(records)
    print(f"  US Census: {len(result)} unique surnames")
    return result


def normalize_spain_ine() -> tuple[pd.DataFrame, pd.DataFrame]:
    first_names = pd.DataFrame()
    surnames = pd.DataFrame()

    for path, label, is_surname in [
        (RAW_DIR / "ine_spain_nombres.xls", "first names", False),
        (RAW_DIR / "ine_spain_apellidos.xls", "surnames", True),
    ]:
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        try:
            import xlrd
            wb = xlrd.open_workbook(str(path))
            records = []
            for sheet_idx in range(wb.nsheets):
                ws = wb.sheet_by_index(sheet_idx)
                gender = "" if is_surname else ("M" if sheet_idx == 0 else "F")
                for i in range(ws.nrows):
                    try:
                        row_vals = [ws.cell_value(i, j) for j in range(min(4, ws.ncols))]
                        name_val = freq_val = None
                        for v in row_vals:
                            if isinstance(v, str) and len(v) >= 2 and not v.replace(".", "").replace(",", "").isdigit():
                                if v.strip().lower() not in ("orden", "nombre", "apellido", "frecuencia", "edad media (*)", ""):
                                    name_val = v
                            elif isinstance(v, (int, float)) and v > 10:
                                if freq_val is None or v > freq_val:
                                    freq_val = v
                        if name_val and freq_val:
                            cleaned = clean_name(name_val)
                            if cleaned and len(cleaned) >= 2:
                                records.append({"name": cleaned, "country": "hispanic", "frequency": int(freq_val),
                                                "gender": gender, "type": "surname" if is_surname else "first_name"})
                    except Exception:
                        continue
            if records:
                df = pd.DataFrame(records).groupby(["name", "country", "type"], as_index=False).agg(
                    {"frequency": "sum", "gender": "first"})
                print(f"  Spain INE: {len(df)} unique {label}")
                if is_surname:
                    surnames = df
                else:
                    first_names = df
        except Exception as e:
            print(f"  [ERROR] Spain INE {label}: {e}")

    return first_names, surnames


def normalize_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Normalizing census datasets...")

    brasil = normalize_brasil_io()
    ssa = normalize_ssa()
    spain_fn, spain_sn = normalize_spain_ine()
    us_census = normalize_us_census_surnames()

    first_names = pd.concat([df[["name", "country", "frequency", "gender", "type"]]
                             for df in [brasil, ssa, spain_fn] if not df.empty], ignore_index=True)
    surnames = pd.concat([df for df in [us_census, spain_sn] if not df.empty], ignore_index=True)

    first_names.to_csv(PROCESSED_DIR / "all_first_names.csv", index=False)
    surnames.to_csv(PROCESSED_DIR / "all_surnames.csv", index=False)
    print(f"\nSaved: {len(first_names)} first names, {len(surnames)} surnames")
    return first_names, surnames


def normalize_pt_irn_top_names() -> pd.DataFrame:
    records = []
    for path, gender in [
        (LUSOPHONE_RAW_DIR / "pt_irn_top20_female.csv", "F"),
        (LUSOPHONE_RAW_DIR / "pt_irn_top20_male.csv", "M"),
    ]:
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        df = pd.read_csv(path, comment="#")
        for _, row in df.iterrows():
            name = clean_name(str(row.get("NOME", "")))
            try:
                count = int(float(row.get("TOTAL", 0)))
            except (TypeError, ValueError):
                count = 0
            if name and count > 0:
                records.append({
                    "name": name,
                    "country_class": "pt",
                    "frequency": count,
                    "gender": gender,
                    "type": "given_name",
                    "source": "pt_irn_top20",
                })
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result = result.groupby(["name", "country_class", "type"], as_index=False).agg({
        "frequency": "sum",
        "gender": "first",
        "source": "first",
    })
    print(f"  PT IRN top names: {len(result)} given names")
    return result


def normalize_pt_irn_admissible_names() -> pd.DataFrame:
    path = LUSOPHONE_RAW_DIR / "pt_irn_admissible_names.csv"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return pd.DataFrame()
    df = pd.read_csv(path, comment="#")
    records = []
    for _, row in df.iterrows():
        name = clean_name(str(row.get("name", "")))
        if name and len(name) >= 2:
            records.append({
                "name": name,
                "country_class": "pt",
                "frequency": 1,
                "gender": str(row.get("gender", "")),
                "type": "given_name",
                "source": "pt_irn_admissible_whitelist",
            })
    result = pd.DataFrame(records)
    print(f"  PT IRN admissible whitelist: {len(result)} given names")
    return result


def normalize_wikidata_lusophone() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    given_rows = []
    surname_rows = []
    people_rows = []
    eval_qids: set[str] = set()

    for path in sorted(LUSOPHONE_RAW_DIR.glob("wikidata_*.csv")):
        df = pd.read_csv(path, comment="#")
        for _, row in df.iterrows():
            qid = str(row.get("qid", "")).strip()
            if not qid:
                continue
            split = split_person_pool(qid)
            label = clean_optional_name(row.get("label", ""))
            given = clean_optional_name(row.get("given_name", ""))
            family = clean_optional_name(row.get("family_name", ""))
            given, family = fallback_given_family(label, given, family)
            country = str(row.get("country", "")).strip()
            country_class = str(row.get("country_class", "")).strip()
            people_rows.append({
                "qid": qid,
                "full_name": label,
                "given_name": given,
                "family_name": family,
                "country": country,
                "country_class": country_class,
                "split": split,
            })
            if split == "eval":
                eval_qids.add(qid)
                continue
            if given and len(given) >= 2 and not given.startswith("Q"):
                given_rows.append({
                    "name": given,
                    "country_class": country_class,
                    "frequency": 1,
                    "gender": "",
                    "type": "given_name",
                    "source": "wikidata_train_pool",
                    "qid": qid,
                    "full_name": label,
                })
            if family and len(family) >= 2 and not family.startswith("Q"):
                surname_rows.append({
                    "name": family,
                    "country_class": country_class,
                    "frequency": 1,
                    "gender": "",
                    "type": "surname",
                    "source": "wikidata_train_pool",
                    "qid": qid,
                    "full_name": label,
                })

    eval_path = LUSOPHONE_RAW_DIR / "eval_pool_qids.txt"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        for qid in sorted(eval_qids):
            f.write(f"{qid}\n")

    given = pd.DataFrame(given_rows)
    surnames = pd.DataFrame(surname_rows)
    people = pd.DataFrame(people_rows)
    print(
        "  Wikidata train-pool tokens: "
        f"{len(given)} given rows, {len(surnames)} surname rows; "
        f"eval-pool QIDs: {len(eval_qids)}"
    )
    return given, surnames, people


def normalize_lusophone() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Normalizing Lusophone country-attribution datasets...")

    pt_top = normalize_pt_irn_top_names()
    pt_whitelist = normalize_pt_irn_admissible_names()
    wd_given, wd_surnames, wd_people = normalize_wikidata_lusophone()

    given_names = pd.concat(
        [df for df in [pt_top, pt_whitelist, wd_given] if not df.empty],
        ignore_index=True,
    )
    surnames = wd_surnames

    if not given_names.empty:
        given_names.to_csv(PROCESSED_DIR / "lusophone_given_names.csv", index=False)
    if not surnames.empty:
        surnames.to_csv(PROCESSED_DIR / "lusophone_surnames.csv", index=False)
    if not wd_people.empty:
        wd_people.to_csv(PROCESSED_DIR / "lusophone_wikidata_people.csv", index=False)

    print(f"\nSaved Lusophone normalized data: {len(given_names)} given rows, {len(surnames)} surname rows")
    return given_names, surnames, wd_people


if __name__ == "__main__":
    normalize_all()
