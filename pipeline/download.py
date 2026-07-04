"""Download all census datasets for the name classifier."""

import argparse
import csv
import gzip
import re
import shutil
import time
import zipfile
from pathlib import Path

import requests

from config import (
    RAW_DIR, BRASIL_IO_URL, BRASIL_IO_FALLBACK_URL, SSA_NAMES_URL,
    US_CENSUS_SURNAMES_URL, INE_SPAIN_FIRST_NAMES_URL, INE_SPAIN_SURNAMES_URL,
    LUSOPHONE_RAW_DIR, PT_IRN_FEMALE_TOP_NAMES_URL, PT_IRN_MALE_TOP_NAMES_URL,
    PT_IRN_ADMISSIBLE_NAMES_PDF_URL, WIKIDATA_SPARQL_URL, WIKIDATA_PAGE_SIZE,
    WIKIDATA_MAX_ROWS_PER_COUNTRY, WIKIDATA_BACKOFF_SECONDS, WIKIDATA_USER_AGENT,
    LUSOPHONE_COUNTRIES, WIKIDATA_COUNTRY_CRAWL_ORDER,
)
from pipeline.normalize import clean_name

HEADERS = {"User-Agent": "BR-Name-Classifier/1.0 (research project)"}
TIMEOUT = 60


def download_file(url: str, dest: Path, description: str) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {description} -> {dest.name}")
        return True
    print(f"  [downloading] {description}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  [done] {description} ({dest.stat().st_size / (1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"  [ERROR] {description}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_brasil_io() -> bool:
    print("\n=== Brazilian First Names (Brasil.IO / IBGE) ===")
    dest = RAW_DIR / "brasil_io_nomes.csv"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] -> {dest.name}")
        return True
    gz_dest = RAW_DIR / "brasil_io_nomes.csv.gz"
    if download_file(BRASIL_IO_URL, gz_dest, "Brasil.IO gzipped CSV"):
        try:
            with gzip.open(gz_dest, "rb") as f_in, open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_dest.unlink()
            return True
        except Exception as e:
            print(f"  [ERROR] Extraction failed: {e}")
            if gz_dest.exists():
                gz_dest.unlink()
    print("  [fallback] Trying GitHub mirror...")
    return download_file(BRASIL_IO_FALLBACK_URL, dest, "GitHub mirror CSV")


def download_ssa_names() -> bool:
    print("\n=== US First Names (SSA Baby Names) ===")
    ssa_dir = RAW_DIR / "ssa_names"
    marker = ssa_dir / "_download_complete"
    if marker.exists():
        print(f"  [cached] {len(list(ssa_dir.glob('yob*.txt')))} yearly files")
        return True
    zip_dest = RAW_DIR / "ssa_names.zip"
    if not download_file(SSA_NAMES_URL, zip_dest, "SSA baby names ZIP"):
        return False
    try:
        ssa_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_dest, "r") as zf:
            for name in zf.namelist():
                if name.startswith("yob") and name.endswith(".txt"):
                    zf.extract(name, ssa_dir)
        zip_dest.unlink()
        marker.touch()
        print(f"  [extracted] {len(list(ssa_dir.glob('yob*.txt')))} yearly files")
        return True
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return False


def download_us_census_surnames() -> bool:
    print("\n=== US Surnames (Census 2010) ===")
    dest = RAW_DIR / "us_census_surnames.csv"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] -> {dest.name}")
        return True
    zip_dest = RAW_DIR / "us_census_surnames.zip"
    if not download_file(US_CENSUS_SURNAMES_URL, zip_dest, "US Census surnames ZIP"):
        return False
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                print("  [ERROR] No CSV found in ZIP")
                return False
            with zf.open(csv_names[0]) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        zip_dest.unlink()
        return True
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return False


def download_spain_ine() -> bool:
    print("\n=== Spanish Names (INE Spain) ===")
    ok = download_file(INE_SPAIN_FIRST_NAMES_URL, RAW_DIR / "ine_spain_nombres.xls", "Spain INE first names")
    ok &= download_file(INE_SPAIN_SURNAMES_URL, RAW_DIR / "ine_spain_apellidos.xls", "Spain INE surnames")
    return ok


def write_provenance_csv(dest: Path, source_url: str, license_name: str, content: bytes) -> None:
    """Write a CSV with comment provenance lines before the source header."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = content.decode("utf-8-sig")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(f"# source_url: {source_url}\n")
        f.write(f"# license: {license_name}\n")
        f.write("# fetched_by: br-name-classifier pipeline.download --only lusophone\n")
        f.write(text)


def download_pt_irn_top_names() -> bool:
    print("\n=== Portugal IRN Top Given Names (dados.gov.pt) ===")
    ok = True
    for gender, url in [
        ("female", PT_IRN_FEMALE_TOP_NAMES_URL),
        ("male", PT_IRN_MALE_TOP_NAMES_URL),
    ]:
        dest = LUSOPHONE_RAW_DIR / f"pt_irn_top20_{gender}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [cached] {gender} -> {dest.name}")
            continue
        print(f"  [downloading] Portugal IRN top-20 {gender} names...")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            write_provenance_csv(dest, url, "cc-by-sa", resp.content)
            print(f"  [done] {gender} ({dest.stat().st_size} bytes)")
        except Exception as e:
            print(f"  [ERROR] Portugal IRN top-20 {gender}: {e}")
            ok = False
    return ok


def parse_pt_admissible_pdf(pdf_path: Path, csv_path: Path) -> bool:
    """Extract a conservative given-name whitelist from the IRN PDF."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        print(f"  [ERROR] pypdf unavailable for IRN PDF parsing: {e}")
        return False

    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    parsed_pages = 0
    names: set[str] = set()
    gender_by_name: dict[str, str] = {}
    name_pattern = re.compile(r"\b(Femininos|Masculinos)\s+([A-Za-zÀ-ÖØ-öø-ÿ'’.-]{2,})")

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        page_names = 0
        for match in name_pattern.finditer(text):
            gender = "F" if match.group(1) == "Femininos" else "M"
            name = clean_name(match.group(2).replace("'", " ").replace("’", " "))
            if 2 <= len(name) <= 40 and name.replace(" ", "").isalpha():
                names.add(name)
                gender_by_name[name] = gender
                page_names += 1
        if page_names:
            parsed_pages += 1

    coverage = parsed_pages / page_count if page_count else 0.0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# source_url: {PT_IRN_ADMISSIBLE_NAMES_PDF_URL}\n")
        f.write("# license: official IRN publication; no explicit open-data license found\n")
        f.write(f"# pdf_pages: {page_count}\n")
        f.write(f"# parsed_pages_with_names: {parsed_pages}\n")
        f.write(f"# parse_coverage_pct: {coverage:.1%}\n")
        writer = csv.DictWriter(f, fieldnames=["name", "gender"])
        writer.writeheader()
        for name in sorted(names):
            writer.writerow({"name": name, "gender": gender_by_name.get(name, "")})
    print(f"  [parsed] {len(names)} admissible names; page coverage {coverage:.1%}")
    return bool(names)


def download_pt_admissible_names_pdf() -> bool:
    print("\n=== Portugal IRN Admissible Given Names PDF ===")
    pdf_path = LUSOPHONE_RAW_DIR / "pt_irn_admissible_names.pdf"
    csv_path = LUSOPHONE_RAW_DIR / "pt_irn_admissible_names.csv"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        if not download_file(
            PT_IRN_ADMISSIBLE_NAMES_PDF_URL,
            pdf_path,
            "Portugal IRN admissible given-name PDF",
        ):
            return False
    else:
        print(f"  [cached] PDF -> {pdf_path.name}")
    if csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"  [cached] whitelist -> {csv_path.name}")
        return True
    return parse_pt_admissible_pdf(pdf_path, csv_path)


def wikidata_query(country_qid: str, limit: int, offset: int) -> str:
    return f"""
SELECT DISTINCT ?person ?personLabel ?givenNameLabel ?familyNameLabel WHERE {{
  ?person wdt:P31 wd:Q5;
          wdt:P27 wd:{country_qid};
          rdfs:label ?personLabel.
  FILTER(LANG(?personLabel) = "en")
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def fetch_wikidata_page(country_qid: str, limit: int, offset: int) -> list[dict[str, str]]:
    headers = {"User-Agent": WIKIDATA_USER_AGENT, "Accept": "application/sparql-results+json"}
    params = {"query": wikidata_query(country_qid, limit, offset), "format": "json"}
    for attempt in range(5):
        try:
            resp = requests.get(WIKIDATA_SPARQL_URL, params=params, headers=headers, timeout=45)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"retryable WDQS {resp.status_code}")
            resp.raise_for_status()
            bindings = resp.json()["results"]["bindings"]
            rows = []
            for binding in bindings:
                rows.append({
                    "person": binding.get("person", {}).get("value", ""),
                    "personLabel": binding.get("personLabel", {}).get("value", ""),
                    "givenNameLabel": binding.get("givenNameLabel", {}).get("value", ""),
                    "familyNameLabel": binding.get("familyNameLabel", {}).get("value", ""),
                })
            return rows
        except Exception as e:
            sleep_for = WIKIDATA_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry] offset {offset}: {e}; sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"WDQS failed after retries for {country_qid} offset {offset}")


def extract_qid(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def download_wikidata_people() -> bool:
    print("\n=== Wikidata Lusophone Citizen Corpora ===")
    ok = True
    for cc in WIKIDATA_COUNTRY_CRAWL_ORDER:
        meta = LUSOPHONE_COUNTRIES[cc]
        dest = LUSOPHONE_RAW_DIR / f"wikidata_{cc}.csv"
        part_dest = LUSOPHONE_RAW_DIR / f"wikidata_{cc}.csv.part"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [cached] {cc} -> {dest.name}")
            continue
        print(f"  [querying] {cc} {meta['label']} ({meta['wikidata_qid']})")
        rows: list[dict[str, str]] = []
        seen_qids: set[str] = set()
        if part_dest.exists() and part_dest.stat().st_size > 0:
            with open(part_dest, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(line for line in f if not line.startswith("#"))
                for row in reader:
                    qid = row.get("qid", "")
                    if qid and qid not in seen_qids:
                        seen_qids.add(qid)
                        rows.append(row)
            print(f"    [resume] loaded {len(rows)} partial rows from {part_dest.name}")
        if len(rows) >= WIKIDATA_MAX_ROWS_PER_COUNTRY:
            print(f"    [cap] partial has {len(rows)} rows; promoting without more WDQS calls")
            write_wikidata_rows(part_dest, rows[:WIKIDATA_MAX_ROWS_PER_COUNTRY], meta)
            part_dest.replace(dest)
            continue
        offset = len(rows)
        try:
            while True:
                if len(rows) >= WIKIDATA_MAX_ROWS_PER_COUNTRY:
                    print(f"    [cap] reached {WIKIDATA_MAX_ROWS_PER_COUNTRY} rows")
                    break
                page = fetch_wikidata_page(meta["wikidata_qid"], WIKIDATA_PAGE_SIZE, offset)
                print(f"    offset {offset}: {len(page)} rows")
                new_rows = []
                for row in page:
                    qid = extract_qid(row.get("person", ""))
                    if not qid or qid in seen_qids:
                        continue
                    seen_qids.add(qid)
                    new_rows.append({
                        "qid": qid,
                        "label": row.get("personLabel", ""),
                        "given_name": row.get("givenNameLabel", ""),
                        "family_name": row.get("familyNameLabel", ""),
                        "country": cc,
                        "country_class": meta["class"],
                    })
                rows.extend(new_rows)
                write_wikidata_rows(part_dest, rows[:WIKIDATA_MAX_ROWS_PER_COUNTRY], meta)
                if len(page) < WIKIDATA_PAGE_SIZE:
                    break
                offset += WIKIDATA_PAGE_SIZE
                time.sleep(WIKIDATA_BACKOFF_SECONDS)
            part_dest.replace(dest)
            print(f"  [done] {cc}: {len(rows)} rows")
        except Exception as e:
            print(f"  [ERROR] Wikidata {cc}: {e}")
            ok = False
    return ok


def write_wikidata_rows(path: Path, rows: list[dict[str, str]], meta: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        f.write("# source_url: https://query.wikidata.org/sparql\n")
        f.write("# license: CC0\n")
        f.write(f"# country_qid: {meta['wikidata_qid']}\n")
        f.write("# note: WDQS label corpus; given/family fields blank when property joins time out\n")
        writer = csv.DictWriter(
            f,
            fieldnames=["qid", "label", "given_name", "family_name", "country", "country_class"],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def download_lusophone() -> dict[str, bool]:
    LUSOPHONE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "pt_irn_top_names": download_pt_irn_top_names(),
        "pt_irn_admissible_names": download_pt_admissible_names_pdf(),
        "wikidata_people": download_wikidata_people(),
    }
    print("\n=== Lusophone Download Summary ===")
    for source, ok in results.items():
        print(f"  {source}: {'OK' if ok else 'FAILED'}")
    return results


def download_all() -> dict[str, bool]:
    """Download all datasets. Returns source -> success mapping."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "brasil_io": download_brasil_io(),
        "ssa_names": download_ssa_names(),
        "us_census_surnames": download_us_census_surnames(),
        "spain_ine": download_spain_ine(),
    }
    print("\n=== Download Summary ===")
    for source, ok in results.items():
        print(f"  {source}: {'OK' if ok else 'FAILED'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["base", "lusophone", "lusophone-wikidata", "all"], default="all")
    args = parser.parse_args()
    if args.only == "base":
        download_all()
    elif args.only == "lusophone":
        download_lusophone()
    elif args.only == "lusophone-wikidata":
        download_wikidata_people()
    else:
        base = download_all()
        lusophone = download_lusophone()
        failed = [k for k, v in {**base, **lusophone}.items() if not v]
        if failed:
            raise SystemExit(f"Failed downloads: {', '.join(failed)}")
