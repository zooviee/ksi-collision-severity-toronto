# KSI Traffic Collision Severity — Toronto
### DAMO-699-5 Capstone | Group 5 | University of Niagara Falls Canada

> **Vision Zero alignment:** Identify environmental, behavioural, and
> infrastructural factors that predict fatal collision outcomes in Toronto
> (2006–2026) and build a validated ML classifier (AUC ≥ 0.75).

---

## Team Members

| Name | Student ID | Stories |
|---|---|---|
| Oluwaseyi Adebanjo Akinsanya | NF1012975 | 1, 6, 11, 12 |
| Paniebi Karis Ovuru | NF1019078 | 4, 8, 13 |
| Abigia Gashahun Edossa | NF1021072 | 3, 7, 10 |
| Martins Okpako Agomavroghene | NF1013669 | 2, 5, 9, 11 |

**Supervisor:** Dr. Bilal El Toufaili

---

## Project Structure

```
ksi-collision-severity-toronto/
├── src/
│   ├── data_preparation.py        ← Story 1
│   ├── eda_visualizations.py      ← Story 2
│   ├── hypothesis_testing.py      ← Story 3
│   ├── logistic_regression.py     ← Story 4
│   ├── ml_logistic_baseline.py    ← Story 6
│   ├── ml_decision_tree_rf.py     ← Story 7
│   ├── ml_xgboost_shap.py         ← Story 8
│   ├── model_selection_cv.py      ← Story 9
│   ├── geospatial_analysis.py     ← Story 10
|   ├── dashboard.py               ← Story 11
│   └── app.py                     ← Story 12 (Streamlit)
│
├── tests/
│   ├── test_data_preparation.py       ← Story 1  (53 tests, TDD)
│   ├── test_eda_visualizations.py     ← Story 2  (7 tests)
│   ├── test_ml_logistic_baseline.py   ← Story 6  (18 tests, TDD red-green-blue)
│   └── test_app_edge_cases.py         ← Story 12 (5 edge case tests)
│
├── conftest.py                    ← pytest path config (adds src/ to sys.path)
├── data/                          ← Raw CSV (not committed to git)
│   └── .gitkeep
├── outputs/
│   └── story-1/ … story-11/      ← Generated artefacts (not committed)
├── dashboard/                     ← Story 12 React dashboard
│   ├── src/App.jsx
│   ├── public/ksi_data.json
│   └── vercel.json
├── requirements.txt
├── requirements.lock
└── .gitignore
```

---

## Story Map & Dependencies

```
RAW CSV (Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv)
    │
    ▼
Story 1 ──► ksi_encoded.csv
    │
    ├──► Story 2 (EDA)
    ├──► Story 4 (Logistic regression — inference)
    │
    ▼
Story 6 ──► train_indices.csv, test_indices.csv, logistic_baseline_model.pkl
    │
    ▼
Story 7 ──► dt_model.pkl, rf_model.pkl
    │
    ▼
Story 8 ──► xgb_model.pkl, shap_values.npy
    │
    ▼
Story 9 ──► best_model.pkl
    │
    ├──► Story 11 (Streamlit app)
    └──► Story 12 (React dashboard)

Story 3  ── reads raw CSV (needs original categorical columns)
Story 5  ── Word document, no code
Story 10 ── reads raw CSV (needs coordinates and ward names)
```

### Input file rules — why each story reads what it reads

| Story | Input | Reason |
|---|---|---|
| 1 | Raw CSV | Source of truth — cleans and encodes everything |
| 2 | `ksi_encoded.csv` | Needs `acclass_binary` and OHE columns for plots |
| 3 | Raw CSV | Chi-square tests need original columns (`light`, `rdsfcond`, `drivcond`, `traffictl`) — OHE drops these |
| 4 | `ksi_encoded.csv` | Needs OHE dummies for VIF and GLM logit |
| 6 | `ksi_encoded.csv` | Single source of truth for train/test split and SMOTE |
| 7–9 | `ksi_encoded.csv` + Story 6 indices | Inherit the exact same split from Story 6 |
| 10 | Raw CSV | Needs `latitude`, `longitude`, `stname1`, `stname2`, `wardname` |
| 11 | `best_model.pkl` | Prediction only — no raw data needed |

---

## Story Descriptions

| Story | Script | Tasks | What it does |
|---|---|---|---|
| 1 | `data_preparation.py` | #1–11 | Clean, impute (incl. invage cap >110), OHE encode, engineer temporal features, encode target. **No split or SMOTE.** |
| 2 | `eda_visualizations.py` | #12–16 | Severity distribution, categorical distributions, temporal trends, summary stats, stacked proportions |
| 3 | `hypothesis_testing.py` | #17–22 | Chi-square H1–H4, Cramér's V, odds ratios, hypothesis summary table |
| 4 | `logistic_regression.py` | #24–27 | Statsmodels GLM logit, VIF reduction, coefficient table, OR forest plot, plain-language interpretations |
| 5 | `Story5_Research_Questions.docx` | — | Research question refinement, team sign-off |
| 6 | `ml_logistic_baseline.py` | #34–37 | **80/20 stratified split**, **SMOTE** (training only), sklearn LR baseline, C tuning, ROC, confusion matrix |
| 7 | `ml_decision_tree_rf.py` | TBD | Decision Tree + Random Forest, GridSearchCV, RandomizedSearchCV, feature importances, learning curves |
| 8 | `ml_xgboost_shap.py` | TBD | XGBoost with Optuna tuning, SHAP beeswarm/bar/dependence plots |
| 9 | `model_selection_cv.py` | TBD | 10-fold CV all 4 models, model selection rationale, joblib save, pipeline docs |
| 10 | `geospatial_analysis.py` | TBD | KDE heatmap, ward choropleth (fatality rate), top 10 intersections, data recency docs |
| 11 | `app.py` | TBD | Streamlit prediction app with SHAP explanations and colour-coded gauge |
| 12 | `dashboard/` | TBD | React cross-filter dashboard deployed on Vercel |

---

## Output Naming Convention

All output files follow `task_N_description.ext`:

| Story | Task range | Example outputs |
|---|---|---|
| 1 | — | `variable_catalogue.csv`, `missingness_report.csv`, `ksi_encoded.csv` |
| 2 | #12–16 | `task_12_acclass_distribution.png`, `task_16_road_user_fatal_nonfatal_stacked.png` |
| 3 | #17–22 | `task_17_h1_light_rdsfcond.png`, `task_22_hypothesis_results.csv` |
| 4 | #24–27 | `task_24_logistic_model_summary.txt`, `task_26_vif_before_after.png`, `task_27_top_predictors_or_plot.png` |
| 6 | #34–37 | `task_34_roc_curve_logistic.png`, `task_37_model_comparison_table.png` |
| 7+ | TBD | Follow the same `task_N_` prefix convention |

---

## Test Coverage

| Story | Test file | Tests | Approach |
|---|---|---|---|
| 1 | `test_data_preparation.py` | 53 | TDD-inspired, synthetic fixture, all pipeline steps covered |
| 2 | `test_eda_visualizations.py` | 7 | Test-after, matches teammate function names |
| 6 | `test_ml_logistic_baseline.py` | 18 | **TDD red-green-blue** — genuine RED states committed and pushed |
| 11 | `test_app_edge_cases.py` | 5 | Edge case tests, 5 contrasting prediction scenarios |

### TDD evidence for Story 6

Two tests had genuine RED states with committed proof:

**Test 1** (`test_train_size_approximately_80_pct`):
- RED — `ModuleNotFoundError`: module not importable
- GREEN — `load_and_split` implemented, test passes
- BLUE — full parameter/return docstring added

**Test 2** (`test_auc_above_random`):
- RED — synthetic fixture had no signal, AUC = 0.37 < 0.5
- GREEN — fixture fixed with learnable signal (`older_adult` + `hour<6`), AUC = 0.88
- BLUE — docstring added to `_make_encoded_df` explaining the signal

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/zooviee/ksi-collision-severity-toronto.git
cd ksi-collision-severity-toronto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run all tests
pytest tests/ -v

# 3. Run stories in order

# Story 1 — data preparation (run this first)
python src/data_preparation.py \
    --input      data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-1

# Story 2 — EDA (reads ksi_encoded.csv)
python src/eda_visualizations.py \
    --input      outputs/story-1/ksi_encoded.csv \
    --output-dir outputs/story-2

# Story 3 — hypothesis testing (reads raw CSV)
python src/hypothesis_testing.py \
    --input      data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-3

# Story 4 — logistic regression inference (reads ksi_encoded.csv)
python src/logistic_regression.py \
    --input      outputs/story-1/ksi_encoded.csv \
    --output-dir outputs/story-4

# Story 6 — ML baseline (single source of truth for split + SMOTE)
python src/ml_logistic_baseline.py \
    --input      outputs/story-1/ksi_encoded.csv \
    --output-dir outputs/story-6

# Story 7 — Decision Tree + Random Forest (depends on Story 6)
python src/ml_decision_tree_rf.py \
    --input       outputs/story-1/ksi_encoded.csv \
    --output-dir  outputs/story-7 \
    --indices-dir outputs/story-6 \
    --models-dir  outputs/story-6

# Story 8 — XGBoost + SHAP (depends on Stories 6 and 7)
python src/ml_xgboost_shap.py \
    --input       outputs/story-1/ksi_encoded.csv \
    --output-dir  outputs/story-8 \
    --indices-dir outputs/story-6 \
    --models-dir  outputs/story-6 outputs/story-7

# Story 9 — Model selection + CV (depends on Stories 6, 7, 8)
python src/model_selection_cv.py \
    --input       outputs/story-1/ksi_encoded.csv \
    --output-dir  outputs/story-9 \
    --indices-dir outputs/story-6 \
    --models-dir  outputs/story-6 outputs/story-7 outputs/story-8

# Story 10 — Geospatial analysis (reads raw CSV)
python src/geospatial_analysis.py \
    --input      data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-10

# Story 11 — Streamlit app (auto-discovers best_model.pkl)
streamlit run src/app.py

# Story 12 — React dashboard
cd dashboard && npm install && npm run dev
# Deployed at: https://ksi-dashboard.vercel.app
```

---

## Story 1 Outputs

Story 1 produces exactly three files — no more:

| File | Description |
|---|---|
| `variable_catalogue.csv` | 24 key variable definitions, types, nullability |
| `missingness_report.csv` | Full-column missingness audit (count + %) |
| `ksi_encoded.csv` | Cleaned, encoded dataset — all 20,439 KSI records |

**Story 1 does NOT produce split or SMOTE files.**
Train/test split and SMOTE live exclusively in Story 6.

---

## Story 6 Outputs

Story 6 is the single source of truth for the ML train/test split:

| File | Used by |
|---|---|
| `train_indices.csv` | Stories 7, 8, 9 via `--indices-dir` |
| `test_indices.csv` | Stories 7, 8, 9 via `--indices-dir` |
| `logistic_baseline_model.pkl` | Stories 7, 8 via `--models-dir` |
| `task_34_logistic_baseline_metrics.csv` | Story 9 comparison table |
| `model_comparison_table.csv` | Stories 7, 8, 9 append rows to this |

---

## Known Limitations

| Limitation | Where documented |
|---|---|
| 2024–2026 data underrepresented (police reporting lag 6–18 months) | Story 10 `data_recency_limitation.txt` |
| `aggressive` and `distracted` flags have negative SHAP — reporting bias artefact (flags more common in non-fatal collisions where driver survives to be interviewed) | Story 8 SHAP plots, Story 4 coefficient table |
| `invage` outliers (years recorded as age, e.g. 2023) — 2 records capped at 110 in Story 1 | Story 1 `impute_and_flag()` |
| XGBoost trained on Toronto KSI data only — not validated for other cities or jurisdictions | Story 9 pipeline docs |
| Streamlit app not for operational use — probabilistic population-level estimates only | Story 11 disclaimer |

---

## Data Source

City of Toronto Open Data Portal —
[Motor Vehicle Collisions with KSI Data](https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/)

- Timeframe: 2006–2026 (daily refresh)
- Records after cleaning: 20,439 (dropped 18 Property Damage Only)
- Fatal: 2,886 (14.1%) | Non-Fatal: 17,553 (85.9%)
- Imbalance ratio: 6.08:1 (handled by SMOTE in Story 6)

---

## Branch Strategy

```
main
└── feature/story-1-data-preparation      ← Oluwaseyi  ✓ merged
└── feature/story-2-eda                   ← Paniebi    (in review)
└── feature/story-3-hypothesis-testing    ← Abigia     (in progress)
└── feature/story-4-logistic              ← Martins    (in progress)
└── feature/story-6-ml-baseline           ← Oluwaseyi  (in review)
└── feature/story-7-dt-rf                 ← Martins    (pending)
└── feature/story-8-xgboost-shap          ← Martins    (pending)
└── feature/story-9-model-selection       ← Oluwaseyi  (pending)
└── feature/story-10-geospatial           ← Abigia     (pending)
└── feature/story-11-streamlit-app        ← Oluwaseyi  (pending)
└── feature/story-12-dashboard            ← Oluwaseyi  (pending)
```

Each story gets its own feature branch. Open a Pull Request into `main`
and require at least 1 team-member review before merging.

---

## Git Workflow — Step by Step

---

### Step 1 — Get the latest `main`

```bash
git checkout main
git pull origin main
```

### Step 2 — Create your feature branch

```bash
git checkout -b feature/story-N-short-description
```

### Step 3 — Do your work

Run the pipeline and tests before committing:

```bash
python src/your_script.py --input ... --output-dir outputs/story-N
pytest tests/ -v
```

### Step 4 — Stage only your story's files

```bash
git status
git add src/your_script.py
git add tests/test_your_script.py
git add README.md   # only if you updated it
```

**Never add:** `outputs/`, `data/`, `.venv/`

### Step 5 — Commit with a detailed message

```bash
git commit -m "feat(story-N): short summary

- What was done
- Key result or metric
- Output files produced"
```

**Commit prefixes:**

| Prefix | When to use |
|---|---|
| `feat(story-N):` | New pipeline or feature |
| `fix(story-N):` | Bug fix |
| `test(story-N): RED` | Failing test — written before code exists |
| `test(story-N): GREEN` | Code written to pass the failing test |
| `refactor(story-N): BLUE` | Refactor — all tests still passing |
| `docs:` | README or documentation only |
| `chore:` | Config, gitignore, tooling |

### Step 6 — Push

```bash
git push -u origin feature/story-N-short-description  # first push
git push                                                # subsequent pushes

# If rejected
git pull origin feature/story-N-short-description --rebase
git push
```

### Step 7 — Open a Pull Request

1. Go to `https://github.com/zooviee/ksi-collision-severity-toronto`
2. Click **Compare & pull request**
3. Use this template:

```
## Summary
One sentence describing what this story implements.

## What this PR includes
- `src/script_name.py`
- `tests/test_script_name.py`

## Output files (task_N naming)
- task_N_filename.png
- task_N_filename.csv

## How to run
python src/script_name.py \
    --input  ... \
    --output-dir outputs/story-N

## Tests
pytest tests/test_script_name.py -v
# Expected: N passed

## Story dependency
Depends on: Story X (which file)
Required by: Story Y

## Checklist
- [ ] Script runs on full dataset without errors
- [ ] All task_N output files saved to outputs/story-N/
- [ ] All tests passing
- [ ] No outputs/ or data/ files committed
- [ ] README updated if run commands changed
```

4. Assign at least 1 reviewer → **Create pull request**

### Step 8 — Respond to review feedback

```bash
git add src/your_script.py
git commit -m "fix(story-N): address PR review — description"
git push
```

### Step 9 — Merge into main

1. Click **Squash and merge**
2. Keep the story prefix in the commit message
3. Delete the feature branch
4. Pull locally:

```bash
git checkout main && git pull origin main
git branch -d feature/story-N-short-description
```

### Quick reference

```bash
git checkout main && git pull origin main
git checkout -b feature/story-N-short-description
# work, test
git add src/script.py tests/test_script.py
git commit -m "feat(story-N): summary"
git push -u origin feature/story-N-short-description
# Open PR → assign reviewer → squash and merge
git checkout main && git pull origin main
git branch -d feature/story-N-short-description
```