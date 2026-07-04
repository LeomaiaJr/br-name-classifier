# BR Name Classifier

ML classifier that identifies Brazilian and broader Lusophone name-origin signals by comparing against public name-frequency datasets from multiple countries. Uses character n-gram patterns + cross-country frequency analysis to score names from 0-100 for Brazilian-likelihood.

## How It Works

The classifier combines three layers:

1. **Census frequency lookup** — compares name against 260K+ names from IBGE (Brazil), US SSA, US Census, and Spain INE. Computes P(Brazilian|name) using Bayesian probability across countries.

2. **Character n-gram model** — a TF-IDF vectorizer (char_wb, 2-4 grams) + SGDClassifier trained on 600K synthetic names. Catches uncommon names not in any census by learning that character patterns like `-eiro`, `-aldo`, `-inha` are distinctly Portuguese.

3. **Meta-classifier** — a logistic regression that combines 27 features (census probabilities, role-aware name slots, n-gram scores, structural signals like Portuguese prepositions DA/DOS, Brazilian suffixes JUNIOR/NETO, Hispanic indicators) into a final P(Brazilian) score.

## Results

| Metric | Value |
|--------|-------|
| N-gram model accuracy | 90.6% (3-class) |
| Meta-classifier accuracy | 99.6% (binary) |
| Uncommon name detection | CLEUDIMAR, GENIVALDO, GIVANILDO all score 100 |
| Hispanic rejection | HERNANDEZ JUAN CARLOS → 0 |
| American rejection | SMITH JOHN → 0 |

## Quick Start

```bash
# Install
pip install -e .

# Download census data (~40 MB)
python -m pipeline.download

# Run full pipeline (normalize → merge → generate training data → train → export)
python -m pipeline.normalize
python -m pipeline.merge
python -m pipeline.generate_training_data
python -m pipeline.train
python -m pipeline.train_meta
python -m pipeline.export

# Score a name
python scorer.py "FERREIRA GUSTAVO DA SILVA"
# Score: 100/100 (Very High)

# Run test suite
python scorer.py
```

## Usage as a Library

```python
from scorer import classify_name, classify_record

# Single name
result = classify_name("GENIVALDO DE SOUZA")
print(result.score)        # 100
print(result.confidence)   # "Very High"
print(result.reasons)      # ["SOUZA: 41% BR surname (census)", ...]
print(result.probabilities) # {"brazilian": 0.999, "hispanic": 0.001, ...}

# Two related name fields
result = classify_record("FERREIRA GUSTAVO", "FERREIRA JULIANA")
print(result.score)  # 100 (bonus for both names matching)
```

## Architecture

```
pipeline/download.py          ← Fetch census data from IBGE, SSA, US Census, Spain INE
pipeline/normalize.py         ← Parse and normalize all datasets
pipeline/merge.py             ← Build cross-country frequency tables
pipeline/generate_training_data.py  ← Generate 600K synthetic names (200K per class)
pipeline/train.py             ← Train character n-gram classifier (TF-IDF + SGD)
pipeline/train_meta.py        ← Train 27-feature meta-classifier
pipeline/export.py            ← Export models to JSON

scorer.py                     ← Scoring API (classify_name, classify_record)
config.py                     ← Paths, URLs, hyperparameters
constants.py                  ← Shared constants (prepositions, suffixes, etc.)

output/
  frequency_tables.json       ← Census name probabilities (6.4 MB)
  ngram_model.json            ← Sparse n-gram model weights (298 KB)
  meta_model.json             ← Meta-classifier weights (682 bytes)
  pcthispanic_lookup.json     ← US Census ethnicity data (1.4 MB)
```

## Data Sources

| Source | Records | What |
|--------|--------:|------|
| [Brasil.IO / IBGE](https://brasil.io/dataset/genero-nomes/nomes/) | 100,787 | Brazilian first names with frequency |
| [US SSA Baby Names](https://www.ssa.gov/oact/babynames/limits.html) | 104,819 | American first names (1880-present) |
| [US Census 2010 Surnames](https://www.census.gov/topics/population/genealogy/data/2010_surnames.html) | 162,254 | US surnames with ethnicity percentages |
| [Spain INE](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736177009) | 82,038 | Spanish first names and surnames |

## The 27 Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | max_surname_br_prob | Highest Brazilian probability among surnames (census) |
| 2 | max_firstname_br_prob | Highest Brazilian probability among first names (census) |
| 3 | max_surname_hispanic_prob | Highest Hispanic probability among surnames (census) |
| 4 | max_firstname_hispanic_prob | Highest Hispanic probability among first names (census) |
| 5 | role_surname_br_prob | Brazilian surname probability in likely surname slots |
| 6 | role_firstname_br_prob | Brazilian first-name probability in likely given-name slots |
| 7 | role_surname_hispanic_prob | Hispanic surname probability in likely surname slots |
| 8 | role_firstname_hispanic_prob | Hispanic first-name probability in likely given-name slots |
| 9 | role_firstname_american_prob | American first-name probability in likely given-name slots |
| 10 | br_surname_with_nonbr_firstname | Strong BR surname paired with non-BR/Anglo given name |
| 11 | last_first_br_signal | Product signal for `LAST FIRST` order |
| 12 | first_last_br_signal | Product signal for natural `FIRST LAST` order |
| 13 | first_last_hispanic_signal | Hispanic product signal for natural `FIRST LAST` order |
| 14 | ngram_br_prob | Brazilian probability from character n-gram model |
| 15 | ngram_hispanic_prob | Hispanic probability from character n-gram model |
| 16 | ngram_american_prob | American probability from character n-gram model |
| 17 | portuguese_preposition_count | Count of DA, DOS, DAS, DO, DE |
| 18 | spanish_preposition_present | DE LOS, DE LA, DEL detected |
| 19 | brazilian_suffix_present | JUNIOR, NETO, FILHO detected |
| 20 | ez_surname_count | Words ending in -EZ (Hispanic indicator) |
| 21 | merged_preposition_detected | DASILVA, DESOUZA etc. |
| 22 | us_census_pcthispanic | US Census Hispanic percentage for surname |
| 23 | surname_in_any_census | Surname found in any census dataset |
| 24 | firstname_in_any_census | First name found in any census dataset |
| 25 | compound_name_detected | ANA PAULA, JOAO PEDRO etc. |
| 26 | name_token_count | Number of words in name |
| 27 | both_names_have_signal | Both NAME1 and NAME2 have Brazilian indicators |

## License

MIT
