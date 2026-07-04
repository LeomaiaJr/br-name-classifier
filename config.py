"""Configuration for BR Name Classifier data pipeline."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LUSOPHONE_RAW_DIR = RAW_DIR / "lusophone"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data source URLs
BRASIL_IO_URL = "https://data.brasil.io/dataset/genero-nomes/nomes.csv.gz"
BRASIL_IO_FALLBACK_URL = "https://raw.githubusercontent.com/datasets-br/prenomes/main/data/nomes-censos-ibge.csv"

SSA_NAMES_URL = "https://www.ssa.gov/oact/babynames/names.zip"

US_CENSUS_SURNAMES_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"

# Spain INE - these are direct download links
INE_SPAIN_FIRST_NAMES_URL = "https://www.ine.es/daco/daco42/nombyapel/nombres_por_edad_media.xls"
INE_SPAIN_SURNAMES_URL = "https://www.ine.es/daco/daco42/nombyapel/apellidos_frecuencia.xls"

# Portugal IRN / dados.gov.pt official annual top-20 baby-name CSV API resources.
# License observed in dados.gov.pt metadata: cc-by-sa.
PT_IRN_FEMALE_TOP_NAMES_URL = "https://dados.gov.pt/api/1/datasets/r/e52dce08-035a-4eb5-bd2a-224f8f11e3a1"
PT_IRN_MALE_TOP_NAMES_URL = "https://dados.gov.pt/api/1/datasets/r/65175021-c8ca-4618-a04e-595c34ba15f6"

# Portugal IRN admissible given-name PDF. Official publication; no explicit
# open-data license found in the source research, so it is used as a whitelist.
PT_IRN_ADMISSIBLE_NAMES_PDF_URL = (
    "https://irn.justica.gov.pt/Portals/33/Regras%20Nome%20Proprio/"
    "Lista%20Nomes%20Pr%C3%B3prios.pdf?ver=WNDmmwiSO3uacofjmNoxEQ%3D%3D"
)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_PAGE_SIZE = 2000
WIKIDATA_MAX_ROWS_PER_COUNTRY = 60_000
WIKIDATA_BACKOFF_SECONDS = 2.0
WIKIDATA_USER_AGENT = (
    "br-name-classifier/1.0 Lusophone classifier research "
    "(Wikidata CC0 attribution corpus)"
)

LUSOPHONE_COUNTRIES = {
    "br": {"label": "Brazil", "wikidata_qid": "Q155", "class": "br"},
    "pt": {"label": "Portugal", "wikidata_qid": "Q45", "class": "pt"},
    "cv": {"label": "Cape Verde", "wikidata_qid": "Q1011", "class": "cv"},
    "ao": {"label": "Angola", "wikidata_qid": "Q916", "class": "ao"},
    "mz": {"label": "Mozambique", "wikidata_qid": "Q1029", "class": "mz"},
    "gw": {"label": "Guinea-Bissau", "wikidata_qid": "Q1007", "class": "palop_other"},
    "st": {"label": "Sao Tome and Principe", "wikidata_qid": "Q1039", "class": "palop_other"},
    "tl": {"label": "Timor-Leste", "wikidata_qid": "Q574", "class": "palop_other"},
}

WIKIDATA_COUNTRY_CRAWL_ORDER = ("cv", "gw", "st", "tl", "mz", "ao", "pt", "br")

LUSOPHONE_CLASSES = ("br", "pt", "cv", "ao", "mz", "palop_other")

LUSOPHONE_UNATTRIBUTABLE_PRIOR = {
    "br": 0.45,
    "pt": 0.25,
    "cv": 0.10,
    "ao": 0.08,
    "mz": 0.07,
    "palop_other": 0.05,
}

# Mexico - original Datamx URLs are dead, skipping for now.
# Hispanic coverage comes from Spain INE + US Census pcthispanic column.
MEXICO_MALES_URL = None
MEXICO_FEMALES_URL = None
MEXICO_SURNAMES_URL = None

# DEPRECATED: frequency tables now use a US-resident-bearer basis for all
# classes (see merge.py BR_US_SCALE / US_HISPANIC_SCALE). Kept for reference.
POPULATION = {
    "brazilian": 210,
    "american": 330,
    "hispanic": 47,  # Spain INE only (Mexico/LatAm sources unavailable)
}

# Training configuration
TRAINING_SAMPLES_PER_CLASS = 200_000
RANDOM_SEED = 42

# N-gram model hyperparameters (defaults, tuned via GridSearchCV)
NGRAM_RANGE = (2, 4)
MAX_FEATURES = 15000
MIN_DF = 2

# Export configuration
FREQUENCY_MIN_OCCURRENCES = 200  # Minimum census occurrences to include in frequency tables
PROBABILITY_DECIMALS = 4  # Round probabilities to this many decimal places
MODEL_VERSION = "2026-07-04"

# Confidence level thresholds
CONFIDENCE_THRESHOLDS = {
    "Very High": 70,
    "High": 50,
    "Medium": 30,
    "Low": 15,
}
