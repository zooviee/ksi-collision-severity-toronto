# KSI Traffic Collision Severity — Toronto
### DAMO-699-5 Capstone | Group 5 | University of Niagara Falls Canada

> **Vision Zero alignment:** Identify environmental, behavioural, and
> infrastructural factors that predict fatal collision outcomes in Toronto
> (2006–2026) and build a validated ML classifier (AUC ≥ 0.75).

---


## Repository Setup (one-time, done by team lead)

```bash
# 1. Create the repo on GitHub (public or private)
#    GitHub → New repository → Name: ksi-collision-severity-toronto
#    Add a README, choose Python .gitignore

# 2. Clone locally
git clone https://github.com/<your-org>/ksi-collision-severity-toronto.git
cd ksi-collision-severity-toronto

# 3. Copy project files in
cp -r /path/to/ksi_project/* .

# 4. Invite collaborators
#    GitHub repo → Settings → Collaborators → Add people
#    Add each team member's GitHub username and set role to "Write"

# 5. Initial commit on main
git add .
git commit -m "chore: initial project scaffold with Story 1 pipeline"
git push origin main
```

---

## Branch Strategy

Each story gets its own feature branch, branched from `main`.

```bash
# Story 1 — Data Preparation (this branch)
git checkout -b feature/story-1-data-preparation
git push -u origin feature/story-1-data-preparation

# When story is complete, open a Pull Request → main
# Require at least 1 team-member review before merging
```

**Branch naming convention:**
```
feature/story-<N>-<short-description>
```

---

## Project Structure

```
ksi-collision-severity-toronto/
├── src/
│   └── data_preparation.py      # Story 1 pipeline (all 10 tasks)
├── tests/
│   └── test_data_preparation.py # TDD test suite (60 tests)
├── data/                        # Raw data — NOT committed to git
│   └── .gitkeep
├── outputs/                     # Pipeline artefacts — NOT committed to git
│   ├── variable_catalogue.csv
│   ├── missingness_report.csv
│   ├── ksi_encoded.csv
│   ├── X_train_smote.csv
│   ├── y_train_smote.csv
│   ├── X_test.csv
│   └── y_test.csv
├── notebooks/                   # EDA notebooks (Story 2)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full Story 1 pipeline
python src/data_preparation.py \
    --input data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs

# Run the test suite
pytest tests/test_data_preparation.py -v
```

---

## Story 1 — Data Preparation Pipeline

**Branch:** `feature/story-1-data-preparation`

### Tasks completed

| # | Task | Status |
|---|---|---|
| 1 | Load dataset & confirm shape (~20 457 rows, ~50 cols) | ✅ |
| 2 | Document all 24 key variable definitions | ✅ |
| 3 | Missingness audit: count + % per column, flag >20% | ✅ |
| 4 | Mode-impute categoricals; flag behavioural variables | ✅ |
| 5 | One-hot encode nominals; label-encode ordinal (injury) | ✅ |
| 6 | Engineer temporal features from accdate | ✅ |
| 7 | Binary-encode target: Fatal=1, Non-Fatal=0 | ✅ |
| 8 | SMOTE on training split only; verify no test leakage | ✅ |
| 9 | Final validation assertions (nulls, bounds, binary) | ✅ |
| 10 | TDD: 60 unit + integration tests, all passing | ✅ |

### Key findings from real data

| Metric | Value |
|---|---|
| Total records | 20 458 |
| KSI records (after dropping 19 PDO rows) | 20 439 |
| Fatal Injury | 2 886 (14.1%) |
| Non-Fatal Injury | 17 553 (85.9%) |
| Class imbalance ratio | 6.08 : 1 |
| Columns >20% missing (flagged) | 15 |
| `drivcond` missing | 9 929 rows (48.5%) — behavioural reporting gap |
| Feature columns for modelling | 94 |
| SMOTE: train set after oversampling | 28 084 (balanced 50/50) |
| Test set size (untouched) | 4 088 |

### Imputation decisions

| Column | Action | Rationale |
|---|---|---|
| `light`, `rdsfcond`, `traffictl`, `road_class`, `impactype` | Mode imputation | <5% missing; mode is sensible default |
| `accloc` | Mode imputation | 26.7% missing — flagged; mode used to preserve rows |
| `invage` | Median imputation | Right-skewed; median robust to outliers |
| `drivcond` | Missingness indicator added, NOT imputed | 48.5% missing — reflects reporting gap, not absence of behaviour |
| `aggressive`, `distracted` | Missingness indicator added, NOT imputed | Behavioural — same rationale |

### Limitation note
> The `drivcond` variable is missing in 48.5% of records. This likely reflects
> incomplete police reporting rather than a true absence of driver condition
> information. Derived statistical estimates from this variable should be
> interpreted with caution. This limitation is noted in the project proposal (§4).

---

## Data Source

City of Toronto Open Data Portal —
[Motor Vehicle Collisions with KSI Data](https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/)

- Timeframe: 2006–2026 (daily refresh)
- Provided by: Toronto Police Service / Transportation Services
- Scope: All KSI collisions within Toronto right-of-way involving ≥1 motor
  vehicle or streetcar; excludes private property and provincial highways.

---

## Testing

```bash
pytest tests/ -v
# Expected: 60 passed
```

Tests use a fully synthetic fixture dataset — no external files required.
Test classes map 1-to-1 with pipeline steps:

- `TestLoadDataset` → Step 1
- `TestVariableCatalogue` → Step 2
- `TestMissingnessAudit` → Step 3
- `TestImputation` → Step 4
- `TestCategoricalEncoding` → Step 5
- `TestTemporalFeatures` → Step 6
- `TestTargetEncoding` → Step 7
- `TestSMOTE` → Step 8
- `TestValidation` → Step 9
- `TestIntegration` → End-to-end smoke test