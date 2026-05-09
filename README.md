# KSI Traffic Collision Severity — Toronto
### DAMO-699-5 Capstone | Group 5 | University of Niagara Falls Canada

> **Vision Zero alignment:** Identify environmental, behavioural, and
> infrastructural factors that predict fatal collision outcomes in Toronto
> (2006–2026) and build a validated ML classifier (AUC ≥ 0.75).

---

## Team Members

| Name | Student ID |
|---|---|
| Oluwaseyi Adebanjo Akinsanya | NF1012975 |
| Paniebi Karis Ovuru | NF1019078 |
| Abigia Gashahun Edossa | NF1021072 |
| Martins Okpako Agomavroghene | NF1013669 |

**Supervisor:** Dr. Bilal El Toufaili

---

## Project Structure

```
ksi-collision-severity-toronto/
├── src/
│   ├── data_preparation.py        ← Story 1
│   ├── eda_visualizations.py      ← Story 2
│   ├── statistical_inference.py   ← Story 3
│   ├── logistic_regression.py     ← Story 4
│   ├── ml_logistic_baseline.py    ← Story 6
│   ├── ml_decision_tree_rf.py     ← Story 7
│   ├── ml_xgboost_shap.py         ← Story 8
│   ├── model_selection_cv.py      ← Story 9
│   ├── geospatial_analysis.py     ← Story 10
│   └── app.py                     ← Story 11 (Streamlit)
│
├── tests/
│   ├── test_data_preparation.py   ← Story 1 TDD (53 tests)
│   └── test_app_edge_cases.py     ← Story 11 edge cases (5 tests)
│
├── data/                          ← Raw CSV (not committed to git)
│   └── .gitkeep
├── outputs/
│   └── story-1/ … story-11/      ← Generated artefacts (not committed)
├── dashboard/                     ← Story 12 React dashboard
│   ├── src/App.jsx
│   ├── public/ksi_data.json
│   └── …
├── requirements.txt
├── requirements.lock
└── .gitignore
```

---

## Story Map & Dependencies

```
Story 1  → ksi_encoded.csv
             ↓               ↓               ↓
           Story 2         Story 3         Story 4
           (EDA)           (Inference)     (Logit inference)

Story 1 → ksi_encoded.csv
             ↓
           Story 6  → train_indices.csv, test_indices.csv, logistic_baseline_model.pkl
             ↓
           Story 7  → dt_model.pkl, rf_model.pkl
             ↓
           Story 8  → xgb_model.pkl, shap_values.npy
             ↓
           Story 9  → best_model.pkl (joblib)
             ↓
           Story 11 (Streamlit app)
           Story 12 (React dashboard)

Story 10 — fully independent, reads raw CSV directly
Story 5  — Word document, no code dependencies
```

### Key dependency rule
> **Stories 2–4** consume `ksi_encoded.csv` from Story 1.
> **Stories 7–9** consume the train/test index files from Story 6.
> Always run stories in order within each branch.

---

## Story Descriptions

| Story | Script | What it does |
|---|---|---|
| 1 | `data_preparation.py` | Clean data, encode variables, engineer features, validate. Outputs `ksi_encoded.csv`. **No split or SMOTE here.** |
| 2 | `eda_visualizations.py` | Distribution plots, temporal trends, stacked bar charts |
| 3 | `statistical_inference.py` | Chi-square tests (H1–H4), odds ratios, hypothesis summary table |
| 4 | `logistic_regression.py` | Statsmodels logistic regression for **statistical inference** (not ML prediction) |
| 5 | `Story5_Research_Questions.docx` | Refined research questions, team sign-off |
| 6 | `ml_logistic_baseline.py` | **80/20 stratified split**, **SMOTE** (training only), sklearn LR baseline ML classifier |
| 7 | `ml_decision_tree_rf.py` | Decision Tree + Random Forest, GridSearchCV, learning curves |
| 8 | `ml_xgboost_shap.py` | XGBoost with Optuna tuning, SHAP plots |
| 9 | `model_selection_cv.py` | 10-fold CV all 4 models, model selection, joblib save, pipeline docs |
| 10 | `geospatial_analysis.py` | KDE heatmap, ward choropleth, top 10 intersections |
| 11 | `app.py` | Streamlit prediction app with SHAP explanations |
| 12 | `dashboard/` | React cross-filter dashboard (deployed on Vercel) |

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/<org>/ksi-collision-severity-toronto.git
cd ksi-collision-severity-toronto
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
pip install -r requirements.txt

# 2. Run tests
pytest tests/ -v

# 3. Run stories in order
python src/data_preparation.py \
    --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-1

python src/eda_visualizations.py \
    --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-2

python src/statistical_inference.py \
    --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-3

python src/logistic_regression.py \
    --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-4

# Story 6 — creates train/test split and SMOTE (Stories 7–9 depend on this)
python src/ml_logistic_baseline.py \
    --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-6

python src/ml_decision_tree_rf.py \
    --input       data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir  outputs/story-7 \
    --indices-dir outputs/story-6 \
    --models-dir  outputs/story-6

python src/ml_xgboost_shap.py \
    --input       data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir  outputs/story-8 \
    --indices-dir outputs/story-6 \
    --models-dir  outputs/story-6 outputs/story-7

python src/model_selection_cv.py \
    --input       data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir  outputs/story-9 \
    --indices-dir outputs/story-6 \
    --models-dir  outputs/story-6 outputs/story-7 outputs/story-8

python src/geospatial_analysis.py \
    --input      data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-10

# Story 11 — Streamlit app
streamlit run src/app.py
```

---

## Story 1 Outputs

Story 1 produces exactly three files:

| File | Description |
|---|---|
| `variable_catalogue.csv` | 24 key variable definitions, types, nullability |
| `missingness_report.csv` | Full-column missingness audit (count + %) |
| `ksi_encoded.csv` | Cleaned, encoded dataset — all 20,439 KSI records |

**Story 1 does NOT produce:** `X_train_smote.csv`, `X_test.csv`, `y_train_smote.csv`,
`y_test.csv` — these are created in Story 6 where the ML pipeline begins.

---

## Data Source

City of Toronto Open Data Portal —
[Motor Vehicle Collisions with KSI Data](https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/)

- Timeframe: 2006–2026 (daily refresh)
- Records: 20,439 (after removing 18 Property Damage Only records)
- Fatal: 2,886 (14.1%) | Non-Fatal: 17,553 (85.9%)

---

## Branch Strategy

```
main
└── feature/story-1-data-preparation   ← Oluwaseyi
└── feature/story-2-eda                ← Paniebi
└── feature/story-3-inference          ← Abigia
└── feature/story-4-logistic           ← Martins
└── feature/story-6-ml-baseline        ← (assigned)
...
```

Each story gets its own feature branch. Open a Pull Request into `main`
and require at least 1 team-member review before merging.

---

## Git Workflow — Step by Step

Every team member follows this exact sequence for their story.
The example below uses Story 2 — replace the story number and description for your own.

---

### Step 1 — Get the latest `main` before you start

Always start from an up-to-date `main` so your branch does not fall behind.

```bash
git checkout main
git pull origin main
```

---

### Step 2 — Create your feature branch

Branch names follow the pattern: `feature/story-N-short-description`

```bash
git checkout -b feature/story-2-eda-visualizations
```

You are now on your own branch. Any commits you make here do not
affect `main` or your teammates' branches.

---

### Step 3 — Do your work

Write your code, run the pipeline, check the outputs:

```bash
python src/eda_visualizations.py \
    --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
    --output-dir outputs/story-2
```

Run any tests relevant to your story:

```bash
pytest tests/ -v
```

---

### Step 4 — Stage only your story's files

**Do not stage the entire project.** Only add files that belong to your story.
Use `git status` first to see what has changed:

```bash
git status
```

Then add only your files:

```bash
# Your script
git add src/eda_visualizations.py

# Your test file (if you wrote one)
git add tests/test_eda_visualizations.py

# README or requirements if you changed them
git add README.md
```

**Never add:**
```bash
# DO NOT add these — they are generated and excluded by .gitignore
git add outputs/           # ← generated artefacts
git add data/              # ← raw dataset (too large, private)
git add .venv/             # ← virtual environment
```

Confirm exactly what you are about to commit:

```bash
git status
git diff --staged
```

---

### Step 5 — Commit with a detailed message

A good commit message has three parts:
1. **Subject line** — short summary using the `feat(story-N):` prefix
2. **Blank line**
3. **Body** — bullet points describing what was done, why, and key results

```bash
git commit -m "feat(story-2): EDA visualizations pipeline

- Distribution plots for all 24 key variables (Fig 1–5)
- Temporal trend: fatal collisions 2006–2023 with COVID dip annotated (Fig 6)
- Stacked bar charts: fatality by lighting, road surface, road user (Fig 7–9)
- Hourly fatality rate bar chart — peak at 05:00 (26.3% fatal rate) (Fig 10)
- Seasonal breakdown: Summer highest at 15.6% fatality rate (Fig 11)
- Data recency note: 2024–2026 excluded from trend due to police reporting lag
- All figures saved to outputs/story-2/ as PNG (150 dpi)
- Script accepts --input and --output-dir CLI arguments
- Runs in ~45 seconds on full 20,439-record dataset"
```

**Commit message conventions used in this project:**

| Prefix | When to use |
|---|---|
| `feat(story-N):` | New story pipeline or feature |
| `fix(story-N):` | Bug fix in an existing script |
| `test(story-N):` | Adding or fixing test files |
| `docs:` | README or documentation only |
| `refactor(story-N):` | Code restructure, no behaviour change |

---

### Step 6 — Push your branch to GitHub

The first time you push a new branch, use `-u` to set the upstream:

```bash
git push -u origin feature/story-2-eda-visualizations
```

On subsequent pushes (after more commits on the same branch):

```bash
git push
```

If your push is rejected because the remote has changes you don't have:

```bash
git pull origin feature/story-2-eda-visualizations --rebase
git push
```

---

### Step 7 — Open a Pull Request on GitHub

1. Go to the repository on GitHub:
   `https://github.com/zooviee/ksi-collision-severity-toronto`

2. You will see a yellow banner:
   **"feature/story-2-eda-visualizations had recent pushes — Compare & pull request"**
   Click it.

3. Fill in the Pull Request form:

   **Base branch:** `main`
   **Compare branch:** `feature/story-2-eda-visualizations`

   **Title:**
   ```
   Story 2 — EDA Visualizations Pipeline
   ```

   **Description** (paste this template and fill it in):
   ```
   ## Summary
   Implements Story 2: exploratory data analysis visualizations for the
   KSI collision severity dataset.

   ## What this PR includes
   - `src/eda_visualizations.py` — full EDA pipeline script
   - `tests/test_eda_visualizations.py` — N unit tests (all passing)

   ## Figures produced
   - Fig 1–5: variable distributions
   - Fig 6: fatal collision trend 2006–2023
   - Fig 7–9: fatality by lighting, surface, road user
   - Fig 10: hourly fatality rate
   - Fig 11: seasonal breakdown

   ## Key findings
   - Summer peak: 15.6% fatality rate (highest season)
   - 05:00 hour: 26.3% fatality rate (highest hour)
   - Data-lag note documented for 2024–2026

   ## How to run
   \```bash
   python src/eda_visualizations.py \
       --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
       --output-dir outputs/story-2
   \```

   ## Tests
   \```bash
   pytest tests/test_eda_visualizations.py -v
   # Expected: N passed
   \```

   ## Story dependency
   Depends on: Story 1 (`ksi_encoded.csv`)
   Required by: None (EDA is standalone)

   ## Checklist
   - [ ] Script runs without errors on full dataset
   - [ ] All figures saved to outputs/story-2/
   - [ ] Tests passing
   - [ ] No data files committed
   - [ ] No outputs/ folder committed
   ```

4. On the right panel:
   - **Reviewers** → assign at least 1 teammate
   - **Assignees** → assign yourself
   - **Labels** → select `story-2` (create the label if it does not exist)

5. Click **Create pull request**

---

### Step 8 — Respond to review feedback

Your reviewer may leave comments. To update your PR:

```bash
# Make the requested changes locally
# Then stage and commit
git add src/eda_visualizations.py
git commit -m "fix(story-2): address PR review — fix axis labels on Fig 6"

# Push again — the PR updates automatically
git push
```

Leave a reply on each comment when you have addressed it.
When the reviewer is satisfied they will click **Approve**.

---

### Step 9 — Merge into main (after approval)

Once you have at least 1 approval:

1. On the PR page, click **Squash and merge**
   (this keeps `main` history clean — one commit per story)

2. Confirm the merge commit message — keep the story prefix:
   ```
   feat(story-2): EDA visualizations pipeline (#PR_NUMBER)
   ```

3. Click **Confirm squash and merge**

4. Delete the feature branch when prompted (keeps the repo tidy)

5. Pull the updated `main` locally:
   ```bash
   git checkout main
   git pull origin main
   ```

---

### Quick reference — commands in order

```bash
# 1. Start fresh from main
git checkout main
git pull origin main

# 2. Create your branch
git checkout -b feature/story-N-short-description

# 3. Do your work, then check what changed
git status
git diff --staged

# 4. Stage only your files
git add src/your_script.py
git add tests/test_your_script.py

# 5. Commit with a detailed message
git commit -m "feat(story-N): short summary

- Detail 1
- Detail 2
- Key result or metric"

# 6. Push
git push -u origin feature/story-N-short-description

# 7. Go to GitHub → Compare & pull request → fill in the template → assign reviewer

# 8. After approval → Squash and merge on GitHub

# 9. Clean up locally
git checkout main
git pull origin main
git branch -d feature/story-N-short-description
```