"""Shared constants for Brazilian name classification."""

PT_STRONG_PREPS = frozenset(["DA", "DAS", "DO", "DOS"])
PT_WEAK_PREPS = frozenset(["DE"])
ALL_PT_PREPS = PT_STRONG_PREPS | PT_WEAK_PREPS

ES_PATTERNS = ("DE LOS", "DE LA", "DE LAS", "DEL")

# NOTE: "JR" is deliberately absent — it is the standard AMERICAN suffix
# abbreviation; only the spelled-out Portuguese forms are Brazilian evidence.
BR_SUFFIXES = frozenset(["JUNIOR", "NETO", "FILHO", "SOBRINHO", "SEGUNDO", "TERCEIRO"])

MERGED_PREPS = frozenset([
    "DASILVA", "DESOUZA", "DESOUSA", "DEOLIVEIRA", "DEALMEIDA", "DEJESUS",
    "DOSANTOS", "DOSSANTOS", "DOCARMO", "DEPAULA", "DEFREITAS", "DECASTRO",
    "DELIMA", "DEMOURA", "DEBRITO", "DEARAUJO", "DACOSTA", "DAROCHA",
    "DACRUZ", "DACUNHA",
])

COMPOUND_FIRST_NAMES = frozenset([
    "ANA PAULA", "ANA CAROLINA", "ANA CLAUDIA", "ANA LUCIA", "ANA MARIA",
    "MARIA FERNANDA", "MARIA EDUARDA", "MARIA LUIZA", "MARIA CLARA",
    "MARIA HELENA", "MARIA APARECIDA", "MARIA CONCEICAO",
    "JOSE ROBERTO", "JOSE FRANCISCO",
    "LUIZ FERNANDO", "LUIZ CARLOS", "LUIZ HENRIQUE",
    "JOAO PEDRO", "JOAO VICTOR", "JOAO VITOR", "JOAO PAULO", "JOAO GABRIEL",
    "PEDRO HENRIQUE", "MARCOS ANTONIO", "CARLOS EDUARDO", "CARLOS ALBERTO",
    "PAULO ROBERTO", "PAULO CESAR", "PAULO HENRIQUE",
])

BUSINESS_KEYWORDS = frozenset([
    "LLC", "INC", "CORP", "CORPORATION", "TRUST", "PARTNERSHIP", "ASSOCIATION",
    "HOLDINGS", "PROPERTIES", "INVESTMENTS", "MANAGEMENT", "VENTURES",
    "ENTERPRISES", "COMPANY", "LTD", "LP", "LLP",
    "CITY", "COUNTY", "STATE", "BOARD", "DISTRICT", "AUTHORITY", "CHURCH",
    "IGLESIA", "MINISTRIES", "MINISTRY", "HOMEOWNERS", "HOA", "CONDOMINIUM",
    "CONDO", "BANK", "MORTGAGE", "DEPARTMENT", "DEPT",
])

SUFFIX_STRIP = (
    "ETAL", "ET AL", "AS TRUSTEE", "TRUSTEE", "ESTATE", "LIFE EST",
    "LIFE ESTATE", "TRUST", "TR", "REVOCABLE", "IRREVOCABLE", "LIVING", "FAMILY",
)

CONFIDENCE_THRESHOLDS = {"Very High": 70, "High": 50, "Medium": 30, "Low": 15}

FEATURE_NAMES = (
    "max_surname_br_prob",
    "max_firstname_br_prob",
    "max_surname_hispanic_prob",
    "max_firstname_hispanic_prob",
    "role_surname_br_prob",
    "role_firstname_br_prob",
    "role_surname_hispanic_prob",
    "role_firstname_hispanic_prob",
    "role_firstname_american_prob",
    "br_surname_with_nonbr_firstname",
    "last_first_br_signal",
    "first_last_br_signal",
    "first_last_hispanic_signal",
    "ngram_br_prob",
    "ngram_hispanic_prob",
    "ngram_american_prob",
    "portuguese_preposition_count",
    "spanish_preposition_present",
    "brazilian_suffix_present",
    "ez_surname_count",
    "merged_preposition_detected",
    "us_census_pcthispanic",
    "surname_in_any_census",
    "firstname_in_any_census",
    "compound_name_detected",
    "name_token_count",
    "both_names_have_signal",
    "pt_whitelist_membership",
    "cv_cluster_score",
    "mz_cluster_score",
    "ao_cluster_score",
    "country_specific_token_count",
)

# Mission 2 curated Lusophone country-attribution boosts.
#
# Provenance:
# - docs/research/lusophone_features.md, Builder A2 research, 2026-07-04.
# - Forebears-derived clusters are review seeds only; no Forebears scraping is
#   permitted in the pipeline. Confidence notes below are intentionally honest:
#   they describe relative country signal, not ground-truth population counts.
# - Brazil and Portugal given-name lists are supported by IBGE API examples and
#   Portugal IRN top-name/admissible-list research, respectively.

CURATED_BR_GIVEN_NAMES = {
    # BR-positive vs PT, weakly non-exclusive vs AO/CV.
    "WESLEY", "WANDERSON", "WELLINGTON", "WASHINGTON", "WENDEL",
    "YASMIN", "YURI", "KAIQUE", "KAYKY", "KAYLANE",
    # Productive -SON/-ILSON layer visible in IBGE.
    "EDSON", "ANDERSON", "ROBSON", "EMERSON", "JEFERSON",
    "ADILSON", "EDILSON", "GILSON", "NILSON", "JAILSON",
    # Invented/blended feminine layer; useful against PT classic names.
    "ROSINEIDE", "ROSEMEIRE", "CLEIDE", "CLEONICE", "EDILEUSA",
    "ELIENE", "LUCIENE", "MARLEIDE", "NEIDE", "ROSANGELA",
}

CURATED_BR_COMPOUND_GIVEN_NAMES = {
    # BR-heavy modern/common compounds; not a universal Lusophone signal.
    "ANA PAULA", "ANA CLAUDIA", "MARIA EDUARDA", "MARIA LUIZA",
    "JOAO PEDRO", "JOAO VITOR", "PEDRO HENRIQUE", "LUIZ FERNANDO",
    "PAULO ROBERTO", "CARLOS EDUARDO",
}

CURATED_PT_GIVEN_NAMES = {
    # PT-positive only as a soft feature; many also occur in Brazil/Africa.
    "MARIA", "FRANCISCO", "ALICE", "LEONOR", "MATILDE", "BENEDITA",
    "AFONSO", "JOAO", "TOMAS", "DUARTE", "CONSTANCA", "INES",
    "GONCALO", "LOURENCO", "DINIS", "MADALENA", "MARGARIDA", "MARTIM",
}

CURATED_CV_SURNAMES = {
    # CV cluster: SEMEDO/FORTES/VARELA/DE PINA/DA LUZ strongest in this list;
    # common Portuguese surnames here remain low-confidence unless combined.
    "SEMEDO", "TAVARES", "MONTEIRO", "FURTADO", "EVORA", "BRITO",
    "FORTES", "VARELA", "DE PINA", "DA LUZ", "SANCHES", "DELGADO",
    "MORENO", "BARBOSA", "LOPES", "GOMES", "MENDES",
}

CURATED_CV_GIVEN_NAMES = {
    # Low-confidence heuristic CV given-name seeds pending stronger source.
    "NILTON", "ELTON", "EDMILSON", "AILTON", "HELDER", "HELIO",
    "HELTON", "INDIRA", "NEUSA", "JANIRA", "JANDIRA", "DJON",
    "DJANIRA", "ELVINA", "CATIA",
}

CURATED_AO_TOKENS = {
    # AO-positive Bantu/country-concentrated tokens; Portuguese-looking tokens
    # such as MANUEL/DOMINGOS/PAULO are deliberately excluded as hard boosts.
    "KIALA", "BUNGA", "DALA", "KALUNGA", "KASSOMA", "CHIVUKUVUKU",
    "SAMAKUVA", "MAVINGA", "KANDIMBA", "KAPENDA",
}

CURATED_MZ_SURNAMES = {
    # MZ has the strongest surname-cluster evidence among PALOP countries.
    "SITOE", "SITHOE", "MACAMO", "MONDLANE", "LANGA", "COSSA", "TEMBE",
    "MANJATE", "MACHAVA", "NHANTUMBO", "MACUACUA", "MUCHANGA",
    "MATSINHE", "MUIANGA", "MUNGUAMBE", "BANZE", "BILA", "MANHIQUE",
    "SIMBINE", "GUAMBE", "ZANDAMELA", "MANDLATE", "MAGAIA",
    "VILANCULOS", "FUMO", "MANHICA", "CUMBE",
}

CURATED_LUSOPHONE_BOOSTS = {
    "given_names": {
        "br": {"tokens": CURATED_BR_GIVEN_NAMES, "boost": 1.25, "confidence": "medium"},
        "pt": {"tokens": CURATED_PT_GIVEN_NAMES, "boost": 0.75, "confidence": "low"},
        "cv": {"tokens": CURATED_CV_GIVEN_NAMES, "boost": 0.45, "confidence": "low"},
    },
    "compound_given_names": {
        "br": {"tokens": CURATED_BR_COMPOUND_GIVEN_NAMES, "boost": 0.90, "confidence": "medium"},
    },
    "surnames": {
        "cv": {"tokens": CURATED_CV_SURNAMES, "boost": 1.10, "confidence": "medium"},
        "ao": {"tokens": CURATED_AO_TOKENS, "boost": 1.00, "confidence": "low"},
        "mz": {"tokens": CURATED_MZ_SURNAMES, "boost": 1.35, "confidence": "high"},
    },
}
