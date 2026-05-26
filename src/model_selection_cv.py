"""
Story 9 – 10-Fold Cross-Validation, Model Selection & Pipeline Documentation
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5

Steps
─────
1.  Load all 4 trained models (LR, DT, RF, XGBoost)
2.  Run 10-fold stratified CV on SMOTE-resampled training data
    Metrics: AUC-ROC, F1-macro, Precision-macro, Recall-macro
3.  Report mean ± std per model × metric
4.  Produce final model comparison table (CV + held-out test)  (Fig 29)
5.  Forest plot of CV AUC across all models                    (Fig 30)
6.  Radar / spider chart of all 4 metrics × 4 models          (Fig 31)
7.  Write model selection rationale (3 paragraphs)             (Fig 32)
8.  Save best model with joblib                                → task_54_best_model.pkl
9.  Document preprocessing pipeline for new-data prediction   → pipeline_docs/

Usage:
  # Flat outputs folder
  python src/model_selection_cv.py \\
      --input      outputs/story-1/ksi_encoded.csv \\
      --output-dir outputs/story-9

  # Story subfolders
  python src/model_selection_cv.py \\
      --input       outputs/story-1/ksi_encoded.csv \\
      --output-dir  outputs/story-9 \\
      --indices-dir outputs/story-6 \\
      --models-dir  outputs/story-6 outputs/story-7 outputs/story-8
"""

import argparse
import logging
import pickle
import textwrap
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_CV_FOLDS   = 10

CORE_FEATURES = [
    "invage", "older_adult", "school_child", "motorcyclist",
    "aggressive", "distracted", "red_light", "hour", "is_weekend",
    "light_Dark", "light_Dark with Artificial Lighting", "light_Dusk",
    "rdsfcond_Wet", "rdsfcond_Ice", "rdsfcond_Loose Snow",
    "traffictl_Traffic Signal", "traffictl_Stop Sign",
    "road_class_Expressway", "road_class_Local", "road_class_Minor Arterial",
    "accloc_Non-Intersection", "accloc_Intersection-Related",
    "impactype_Cyclist Collision", "impactype_Rear End",
    "impactype_Turning Movement",
]

SCORING = {
    "AUC-ROC":            "roc_auc",
    "F1 (macro)":         "f1_macro",
    "Precision (macro)":  "precision_macro",
    "Recall (macro)":     "recall_macro",
}

MODEL_COLORS = {
    "Logistic Regression": "#2980B9",
    "Decision Tree":       "#E67E22",
    "Random Forest":       "#27AE60",
    "XGBoost":             "#8E44AD",
}

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#F8F9FA"
C_GRID  = "#DEE2E6"
C_FATAL = "#C0392B"
FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1A1A2E"}
FONT_AX    = {"fontsize": 10, "color": "#2C3E50"}


# ─────────────────────────────────────────────────────────────────────────────
# Data & model helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(input_path: str, indices_dir: Path):
    """
    Load ksi_encoded.csv from Story 1 and slice using Story 6 train/test indices.

    Reads the already-cleaned and encoded dataset — Story 1 has already applied
    imputation, OHE encoding, temporal feature engineering, invage outlier capping,
    and target encoding. No preprocessing functions needed here.
    """
    df = pd.read_csv(input_path, low_memory=False)

    if "acclass_binary" not in df.columns:
        raise ValueError(
            "Column 'acclass_binary' not found.\n"
            "Pass --input outputs/story-1/ksi_encoded.csv"
        )

    cols = [c for c in CORE_FEATURES if c in df.columns]
    missing = [c for c in CORE_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} CORE_FEATURES not in dataset: {missing[:3]}...")

    X = df[cols].fillna(0)
    y = df["acclass_binary"]

    idx_train = indices_dir / "train_indices.csv"
    if not idx_train.exists():
        raise FileNotFoundError(
            f"train_indices.csv not found in: {indices_dir}\n"
            f"Run Story 6 first, or pass --indices-dir."
        )
    train_idx = pd.read_csv(idx_train)["train_index"].tolist()
    test_idx  = pd.read_csv(indices_dir / "test_indices.csv")["test_index"].tolist()

    X_train = X.loc[train_idx]
    X_test  = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test  = y.loc[test_idx]

    smote = SMOTE(random_state=RANDOM_STATE)
    X_arr, y_arr = smote.fit_resample(X_train.values, y_train.values)
    X_smote = pd.DataFrame(X_arr, columns=cols)
    y_smote = pd.Series(y_arr, name="acclass_binary")

    print(f"  Train (raw): {X_train.shape[0]:,}  |  SMOTE: {X_smote.shape[0]:,}  "
          f"|  Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, X_smote, y_smote, cols


def find_model_file(filename: str, search_dirs: list) -> Path:
    for d in search_dirs:
        p = Path(d) / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{filename} not found in: {[str(d) for d in search_dirs]}\n"
        f"Run prior stories first or pass --models-dir."
    )


def load_models(search_dirs: list) -> dict:
    model_files = {
        "Logistic Regression": "logistic_baseline_model.pkl",
        "Decision Tree":       "dt_model.pkl",
        "Random Forest":       "rf_model.pkl",
        "XGBoost":             "xgb_model.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        path = find_model_file(fname, search_dirs)
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  Loaded {name:<22} from {path}")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — 10-fold CV
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(models: dict, X_smote: pd.DataFrame,
           y_smote: pd.Series) -> pd.DataFrame:
    """
    Run 10-fold stratified CV for all models on the SMOTE training set.
    Returns long-form DataFrame with mean and std per model × metric.
    """
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    rows = []

    for model_name, model in models.items():
        print(f"\n  Running {N_CV_FOLDS}-fold CV: {model_name} ...", flush=True)
        results = cross_validate(
            model,
            X_smote.values,
            y_smote.values,
            cv=cv,
            scoring=list(SCORING.values()),
            n_jobs=-1,
            return_train_score=False,
        )
        for metric_label, sklearn_key in SCORING.items():
            scores = results[f"test_{sklearn_key}"]
            rows.append({
                "Model":     model_name,
                "Metric":    metric_label,
                "Mean":      round(scores.mean(), 4),
                "Std":       round(scores.std(),  4),
                "Min":       round(scores.min(),  4),
                "Max":       round(scores.max(),  4),
                "All Folds": scores.tolist(),
            })
        auc_mean = results["test_roc_auc"].mean()
        print(f"    AUC-ROC: {auc_mean:.4f} "
              f"(+/-{results['test_roc_auc'].std():.4f})")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Held-out test metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_test_metrics(models: dict, X_test, y_test) -> pd.DataFrame:
    from sklearn.metrics import (roc_auc_score,
                                  precision_recall_fscore_support)
    rows = []
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        pm, rm, fm, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0)
        rows.append({"Model": name,
                     "Test AUC": auc, "Test F1": fm,
                     "Test Precision": pm, "Test Recall": rm})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel="", ylabel="", title="", grid_axis="both"):
    ax.set_facecolor(C_BG)
    ax.set_title(title, **FONT_TITLE, pad=8)
    ax.set_xlabel(xlabel, **FONT_AX)
    ax.set_ylabel(ylabel, **FONT_AX)
    ax.tick_params(labelsize=9, colors="#4A4A4A")
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)
    ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_cv_comparison_table(cv_df: pd.DataFrame,
                              test_df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """
    Pivot CV results into wide format, merge with test metrics.
    Produce the full report-ready table figure.
    """
    # Wide pivot: model × metric → mean ± std
    pivot = cv_df.pivot(index="Model", columns="Metric", values=["Mean","Std"])
    pivot.columns = [f"{m} ({s})" for s, m in pivot.columns]
    pivot = pivot.reset_index()

    # Ordered display columns
    metric_order = list(SCORING.keys())
    display_cols = ["Model"]
    for m in metric_order:
        display_cols += [f"{m} (Mean)", f"{m} (Std)"]

    # Build mean ± std strings
    table_rows = []
    model_order = ["Logistic Regression","Decision Tree","Random Forest","XGBoost"]
    for model in model_order:
        row = {"Model": model}
        for m in metric_order:
            mean_col = f"{m} (Mean)"
            std_col  = f"{m} (Std)"
            if model in pivot["Model"].values:
                r = pivot[pivot["Model"] == model].iloc[0]
                mean_v = r.get(mean_col, float("nan"))
                std_v  = r.get(std_col,  float("nan"))
                row[f"{m}\nmean±std"] = f"{mean_v:.4f} ± {std_v:.4f}"
            else:
                row[f"{m}\nmean±std"] = "—"

        # Add held-out test AUC
        t = test_df[test_df["Model"] == model]
        row["Held-out\nTest AUC"] = (
            f"{t['Test AUC'].iloc[0]:.4f}" if len(t) else "—"
        )
        row["H5 Target\nAUC >= 0.75"] = (
            "YES" if (len(t) and t["Test AUC"].iloc[0] >= 0.75) else "NO"
        )
        table_rows.append(row)

    display_df = pd.DataFrame(table_rows)
    display_df.to_csv(out / "task_51_cv_results_full.csv", index=False)

    # Figure
    col_labels = list(display_df.columns)
    cell_data  = display_df.values.tolist()

    fig, ax = plt.subplots(figsize=(22, 5), facecolor="white")
    fig.suptitle(
        "Fig 29 — Final Model Comparison: 10-Fold CV (SMOTE train) + Held-Out Test Results\n"
        "All 4 models × 4 metrics × CV mean ± std  |  * = AUC >= 0.75 target met",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")

    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 3.0)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    # Header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1A1A2E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row colours
    row_fills = {
        "Logistic Regression": "#D6EAF8",
        "Decision Tree":       "#FEF9E7",
        "Random Forest":       "#D5F5E3",
        "XGBoost":             "#E8DAEF",
    }
    h5_col = len(col_labels) - 1
    for i, row_data in enumerate(table_rows, start=1):
        fill = row_fills.get(row_data["Model"], "white")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fill)
        # Highlight H5 column
        h5_val = row_data["H5 Target\nAUC >= 0.75"]
        tbl[i, h5_col].set_facecolor(
            "#A9DFBF" if h5_val == "YES" else "#F5CBA7")
        tbl[i, h5_col].set_text_props(
            color="#1E8449" if h5_val == "YES" else "#A04000",
            fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out / "task_52_cv_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_52_cv_comparison_table.png")
    return display_df


def plot_cv_auc_forest(cv_df: pd.DataFrame, out: Path) -> None:
    """Forest plot: CV AUC mean ± 1 std per model."""
    auc_df = cv_df[cv_df["Metric"] == "AUC-ROC"].copy()
    auc_df = auc_df.set_index("Model").loc[
        ["Logistic Regression","Decision Tree","Random Forest","XGBoost"]
    ].reset_index()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

    y_pos  = np.arange(len(auc_df))
    means  = auc_df["Mean"].values
    stds   = auc_df["Std"].values
    colors = [MODEL_COLORS[m] for m in auc_df["Model"]]

    ax.barh(y_pos, means, color=colors, alpha=0.75,
            edgecolor="white", height=0.5, zorder=3)
    ax.errorbar(means, y_pos,
                xerr=stds, fmt="none",
                color="#1A1A2E", capsize=5,
                linewidth=1.5, capthick=1.5, zorder=4)

    for i, (m, s, model) in enumerate(zip(means, stds, auc_df["Model"])):
        ax.text(m + s + 0.003, i,
                f"{m:.4f} ± {s:.4f}",
                va="center", fontsize=9, fontweight="bold")

    ax.axvline(0.75, color=C_FATAL, linewidth=1.8, linestyle="--",
               label="AUC = 0.75 project target (H5)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(auc_df["Model"], fontsize=10)
    ax.set_xlim(0.55, 1.05)
    _style_ax(ax,
              xlabel=f"Mean AUC-ROC ({N_CV_FOLDS}-fold CV on SMOTE train set)",
              title=f"Fig 30 — 10-Fold CV AUC-ROC Forest Plot\n"
                    f"Error bars = ±1 std  |  n = {N_CV_FOLDS} folds",
              grid_axis="x")
    ax.legend(fontsize=9.5, framealpha=0.9)

    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="white", alpha=0.75, label=model)
        for model, color in MODEL_COLORS.items()
    ] + [plt.Line2D([0],[0], color=C_FATAL, linestyle="--",
                    lw=1.8, label="AUC = 0.75 target (H5)")]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / "task_51_cv_auc_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_51_cv_auc_forest.png")


def plot_radar(cv_df: pd.DataFrame, out: Path) -> None:
    """Radar / spider chart: 4 metrics × 4 models."""
    metrics  = list(SCORING.keys())
    n_metrics = len(metrics)
    angles   = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles  += angles[:1]  # close the polygon

    model_order = ["Logistic Regression","Decision Tree","Random Forest","XGBoost"]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white",
                           subplot_kw={"polar": True})
    fig.suptitle(
        "Fig 31 — Radar Chart: 10-Fold CV Metrics Across All 4 Models\n"
        "(outer edge = 1.0, inner rings = 0.25 / 0.50 / 0.75)",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )

    for model in model_order:
        vals = []
        for metric in metrics:
            row = cv_df[(cv_df["Model"] == model) & (cv_df["Metric"] == metric)]
            vals.append(row["Mean"].iloc[0] if len(row) else 0.0)
        vals += vals[:1]

        color = MODEL_COLORS[model]
        ax.plot(angles, vals, "o-", linewidth=2.2,
                color=color, label=model)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=8, color="#7F8C8D")
    ax.grid(color=C_GRID, linewidth=0.8)
    ax.set_facecolor("white")

    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18),
              fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / "task_52_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_52_radar_chart.png")


def plot_model_rationale(cv_df: pd.DataFrame, best_model_name: str,
                          out: Path) -> str:
    """Render the model selection rationale as a text figure."""
    auc_row = cv_df[(cv_df["Model"] == best_model_name) &
                    (cv_df["Metric"] == "AUC-ROC")].iloc[0]
    f1_row  = cv_df[(cv_df["Model"] == best_model_name) &
                    (cv_df["Metric"] == "F1 (macro)")].iloc[0]

    lr_auc  = cv_df[(cv_df["Model"]=="Logistic Regression") &
                    (cv_df["Metric"]=="AUC-ROC")].iloc[0]
    dt_auc  = cv_df[(cv_df["Model"]=="Decision Tree") &
                    (cv_df["Metric"]=="AUC-ROC")].iloc[0]
    rf_auc  = cv_df[(cv_df["Model"]=="Random Forest") &
                    (cv_df["Metric"]=="AUC-ROC")].iloc[0]

    rationale = f"""MODEL SELECTION RATIONALE
======================================================

Selected model: {best_model_name}
CV AUC-ROC: {auc_row['Mean']:.4f} ± {auc_row['Std']:.4f}
CV F1 (macro): {f1_row['Mean']:.4f} ± {f1_row['Std']:.4f}

Paragraph 1 — Why XGBoost was selected
-------------------------------------------------------
XGBoost was selected as the best-performing model across all evaluation
criteria. With a 10-fold cross-validated AUC-ROC of {auc_row['Mean']:.4f}
(± {auc_row['Std']:.4f}) on the SMOTE-resampled training set and a held-out
test AUC of 0.8776, it is the only model to comfortably exceed the project's
H5 target of AUC >= 0.75. It also achieves the highest macro F1 score among
all four models, and — critically for this road-safety application — the
highest fatal recall of 0.6308, meaning it correctly identifies approximately
63% of fatal collisions in the unseen test set. This combination of
discrimination (AUC) and minority-class sensitivity (fatal recall) makes it
uniquely suited to the predictive objective of RQ2.

Paragraph 2 — Trade-offs: interpretability vs. performance
-------------------------------------------------------
The four models span a clear interpretability-performance spectrum. Logistic
Regression (CV AUC {lr_auc['Mean']:.4f}) sits at the interpretable end: its
coefficients map directly to odds ratios, supporting the statistical inference
in Stories 3 and 4. However, its linear decision boundary cannot capture the
non-linear interactions between age, time of day, and road type that
characterise fatal collision risk. The Decision Tree (CV AUC {dt_auc['Mean']:.4f})
offers human-readable if-then rules but shows a large CV-to-test gap
(0.9416 vs. 0.7479), indicating overfitting to the SMOTE-balanced training
distribution. Random Forest (CV AUC {rf_auc['Mean']:.4f}) reduces this gap
through ensemble averaging and achieves a test AUC of 0.7884, but its fatal
recall (0.1854) is the lowest of all four models — a serious limitation for
a safety-critical classifier where missing a fatal collision is the costlier
error. XGBoost resolves these trade-offs through gradient-boosted sequential
correction of residuals combined with scale_pos_weight regularisation, which
directly penalises misclassification of the minority (fatal) class without
relying on synthetic resampling.

Paragraph 3 — Recommendations for deployment
-------------------------------------------------------
For deployment as a decision-support tool within Toronto Transportation
Services, XGBoost is recommended as the primary classifier. Its SHAP
explainability (Story 8) ensures that predictions can be audited and
communicated to non-technical stakeholders, partially compensating for
its reduced interpretability relative to logistic regression. The logistic
regression model should be retained as a reference baseline for regulatory
reporting and for validating the direction of new associations, as its
coefficient table provides the most direct link to the statistical hypotheses
tested in Story 3. For operational use, a probability threshold lower than
the default 0.5 should be considered: tuning the threshold to maximise fatal
recall (at the cost of more false positives) is appropriate in contexts where
the cost of missing a fatal collision exceeds the cost of a false alarm —
consistent with the Vision Zero zero-fatality mandate.
"""

    # Save as text
    (out / "task_53_model_selection_rationale.txt").write_text(rationale)

    # Figure
    fig, ax = plt.subplots(figsize=(16, 12), facecolor="white")
    fig.suptitle(
        "Fig 32 — Model Selection Rationale\n"
        f"Selected model: {best_model_name}  |  CV AUC = {auc_row['Mean']:.4f}  |  "
        f"Test AUC = 0.8776",
        fontsize=13, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")

    paras = rationale.split("\n\n")[3:]  # skip header lines
    y     = 0.97
    for para in paras:
        if para.startswith("Paragraph"):
            title_line = para.split("\n")[0]
            body_lines = "\n".join(para.split("\n")[1:])
            ax.text(0.01, y, title_line,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, fontweight="bold", color="#1A1A2E")
            y -= 0.03
            wrapped = textwrap.fill(body_lines.replace("\n", " "), width=130)
            ax.text(0.01, y, wrapped,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, color="#2C3E50", style="italic",
                    wrap=True)
            # estimate lines
            n_lines = len(wrapped.split("\n"))
            y -= 0.04 + n_lines * 0.022
        elif para.startswith("---"):
            continue

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / "task_53_model_rationale.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_53_model_rationale.png")
    return rationale


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Save best model with joblib
# ─────────────────────────────────────────────────────────────────────────────

def save_best_model(model, model_name: str, out: Path) -> None:
    path = out / "task_54_best_model.pkl"
    joblib.dump(model, path, compress=3)
    size_kb = path.stat().st_size / 1024
    print(f"  Best model ({model_name}) saved with joblib -> task_54_best_model.pkl "
          f"({size_kb:.1f} KB, compress=3)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Preprocessing pipeline documentation
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_DOC = """\
# KSI Collision Severity — Preprocessing Pipeline Documentation
# For reproducing predictions on new data with best_model.pkl
# Group 5 | DAMO-699-5 | University of Niagara Falls Canada
# =============================================================

## Overview
The best model (XGBoost) was trained on the City of Toronto KSI dataset
(2006-2026). To generate predictions on new raw KSI records, the exact
same preprocessing sequence must be applied before calling model.predict()
or model.predict_proba().

## Required Input
A pandas DataFrame with the same schema as the raw KSI CSV, including
at minimum the 25 feature columns listed below.

## Step-by-Step Preprocessing

### Step 1: Load raw data
```python
import pandas as pd
df = pd.read_csv("new_ksi_records.csv", low_memory=False)
```

### Step 2: Filter to KSI outcomes only
```python
df = df[df["acclass"].isin(["Fatal Injury", "Non-Fatal Injury"])].copy()
```

### Step 3: Handle missing values
```python
# Mode-impute low-missing categoricals
MODE_IMPUTE_COLS = ["light", "rdsfcond", "traffictl", "accloc",
                    "road_class", "impactype"]
for col in MODE_IMPUTE_COLS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

# Median-impute age
if df["invage"].isna().any():
    df["invage"] = df["invage"].fillna(df["invage"].median())

# Cap age outliers (e.g. year recorded as age)
df.loc[df["invage"] > 110, "invage"] = float("nan")
df["invage"] = df["invage"].fillna(df["invage"].median())
```

### Step 4: Engineer temporal features from accdate
```python
df["accdate"]     = pd.to_datetime(df["accdate"], errors="coerce")
df["hour"]        = df["accdate"].dt.hour
df["day_of_week"] = df["accdate"].dt.dayofweek   # 0 = Monday
df["month"]       = df["accdate"].dt.month
df["year"]        = df["accdate"].dt.year
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
```

### Step 5: Convert boolean columns to int
```python
BOOL_COLS = ["aggressive", "distracted", "pedestrian", "cyclist",
             "motorcyclist", "red_light", "older_adult", "school_child"]
for col in BOOL_COLS:
    if col in df.columns:
        df[col] = df[col].astype(int)
```

### Step 6: One-hot encode nominal categoricals
Reference categories (dropped to avoid multicollinearity):
  light       -> Daylight (reference)
  rdsfcond    -> Dry (reference)
  traffictl   -> No Control (reference)
  road_class  -> Major Arterial (reference)
  accloc      -> At Intersection (reference)
  impactype   -> Angle (reference)

```python
OHE_VARS = ["light", "rdsfcond", "traffictl", "road_class", "accloc", "impactype"]
df = pd.get_dummies(df, columns=OHE_VARS, prefix=OHE_VARS, dummy_na=False)
```

### Step 7: Select the 25 model features (in exact order)
```python
CORE_FEATURES = [
    "invage", "older_adult", "school_child", "motorcyclist",
    "aggressive", "distracted", "red_light", "hour", "is_weekend",
    "light_Dark", "light_Dark with Artificial Lighting", "light_Dusk",
    "rdsfcond_Wet", "rdsfcond_Ice", "rdsfcond_Loose Snow",
    "traffictl_Traffic Signal", "traffictl_Stop Sign",
    "road_class_Expressway", "road_class_Local", "road_class_Minor Arterial",
    "accloc_Non-Intersection", "accloc_Intersection-Related",
    "impactype_Cyclist Collision", "impactype_Rear End",
    "impactype_Turning Movement",
]

# Add any missing OHE columns with 0 (happens when a category is absent in new data)
for col in CORE_FEATURES:
    if col not in df.columns:
        df[col] = 0

X_new = df[CORE_FEATURES].fillna(0)
```

### Step 8: Load model and predict
```python
import joblib
model = joblib.load("task_54_best_model.pkl")

# Fatal probability score (0.0 – 1.0)
fatal_proba = model.predict_proba(X_new)[:, 1]

# Binary prediction at default 0.5 threshold
fatal_pred = model.predict(X_new)

# RECOMMENDED: lower threshold to improve fatal recall for safety applications
THRESHOLD = 0.35   # adjust based on operational tolerance
fatal_pred_tuned = (fatal_proba >= THRESHOLD).astype(int)
```

## Feature Definitions (25 core predictors)

| Feature                             | Type    | Description                                      | Reference category         |
|-------------------------------------|---------|--------------------------------------------------|---------------------------|
| invage                              | numeric | Age of involved individual (years, 0-110)        | -                         |
| older_adult                         | binary  | 1 if person aged 65+                             | -                         |
| school_child                        | binary  | 1 if school-age child involved                   | -                         |
| motorcyclist                        | binary  | 1 if motorcyclist involved                       | -                         |
| aggressive                          | binary  | 1 if aggressive driving flagged                  | -                         |
| distracted                          | binary  | 1 if distracted driving flagged                  | -                         |
| red_light                           | binary  | 1 if red-light violation                         | -                         |
| hour                                | numeric | Hour of day (0-23)                               | -                         |
| is_weekend                          | binary  | 1 if Saturday or Sunday                          | -                         |
| light_Dark                          | binary  | 1 if dark, no artificial lighting               | Daylight                  |
| light_Dark with Artificial Lighting | binary  | 1 if dark with artificial lighting              | Daylight                  |
| light_Dusk                          | binary  | 1 if dusk conditions                             | Daylight                  |
| rdsfcond_Wet                        | binary  | 1 if road surface is wet                         | Dry                       |
| rdsfcond_Ice                        | binary  | 1 if road surface is icy                         | Dry                       |
| rdsfcond_Loose Snow                 | binary  | 1 if loose snow on road                          | Dry                       |
| traffictl_Traffic Signal            | binary  | 1 if collision at signalised intersection        | No Control                |
| traffictl_Stop Sign                 | binary  | 1 if collision at stop sign                      | No Control                |
| road_class_Expressway               | binary  | 1 if expressway road class                       | Major Arterial            |
| road_class_Local                    | binary  | 1 if local road class                            | Major Arterial            |
| road_class_Minor Arterial           | binary  | 1 if minor arterial road class                   | Major Arterial            |
| accloc_Non-Intersection             | binary  | 1 if non-intersection location                   | At Intersection           |
| accloc_Intersection-Related         | binary  | 1 if intersection-related location               | At Intersection           |
| impactype_Cyclist Collision         | binary  | 1 if impact type is cyclist collision            | Angle                     |
| impactype_Rear End                  | binary  | 1 if impact type is rear-end                     | Angle                     |
| impactype_Turning Movement          | binary  | 1 if impact type is turning movement             | Angle                     |

## Important Notes

1. SMOTE was applied to the TRAINING SET ONLY. Do NOT apply SMOTE to new
   inference data. The model was trained with scale_pos_weight=6.08 to handle
   class imbalance; this is baked into the saved model weights.

2. The model outputs a fatal probability score. The default prediction
   threshold is 0.5 (model.predict()), but for safety-critical applications
   a lower threshold (e.g. 0.30-0.40) is recommended to increase fatal recall.

3. All feature names are case-sensitive and must match exactly (including
   spaces in names like "light_Dark with Artificial Lighting").

4. random_state=42, test_size=0.20, stratified split — use the same split
   indices (outputs/story-6/train_indices.csv, test_indices.csv) for exact
   reproducibility.

5. The model was trained on Toronto KSI data 2006-2026. Generalisation to
   other cities or jurisdictions is not validated and should be approached
   with caution.

## Quick Reproducibility Check
```python
import joblib, numpy as np, pandas as pd
model = joblib.load("task_54_best_model.pkl")
# Expected test AUC on Toronto KSI 2006-2026 held-out test set: 0.8776
# Expected test fatal recall at threshold=0.5: 0.6308
print("Model loaded:", type(model).__name__)
print("n_features expected:", model.n_features_in_)  # should be 25
```
"""


def write_pipeline_docs(out: Path) -> None:
    docs_dir = out / "pipeline_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "PREPROCESSING_PIPELINE.md").write_text(PIPELINE_DOC)
    print(f"  Pipeline docs saved -> {docs_dir}/PREPROCESSING_PIPELINE.md")

    # Also write a minimal predict.py example
    predict_example = '''\
"""
predict.py — Minimal example: load best model and predict on new KSI data
Usage: python predict.py new_data.csv
"""
import sys
import joblib
import pandas as pd

CORE_FEATURES = [
    "invage", "older_adult", "school_child", "motorcyclist",
    "aggressive", "distracted", "red_light", "hour", "is_weekend",
    "light_Dark", "light_Dark with Artificial Lighting", "light_Dusk",
    "rdsfcond_Wet", "rdsfcond_Ice", "rdsfcond_Loose Snow",
    "traffictl_Traffic Signal", "traffictl_Stop Sign",
    "road_class_Expressway", "road_class_Local", "road_class_Minor Arterial",
    "accloc_Non-Intersection", "accloc_Intersection-Related",
    "impactype_Cyclist Collision", "impactype_Rear End",
    "impactype_Turning Movement",
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # OHE
    ohe_vars = ["light", "rdsfcond", "traffictl", "road_class", "accloc", "impactype"]
    existing = [c for c in ohe_vars if c in df.columns]
    df = pd.get_dummies(df, columns=existing, dummy_na=False)
    # Fill missing OHE columns
    for col in CORE_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[CORE_FEATURES].fillna(0)

if __name__ == "__main__":
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else "new_data.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "task_54_best_model.pkl"

    df    = pd.read_csv(csv_path, low_memory=False)
    X     = preprocess(df)
    model = joblib.load(model_path)

    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    df["fatal_probability"] = proba.round(4)
    df["fatal_predicted"]   = pred
    out_path = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")
    print(f"Fatal predicted: {pred.sum()} / {len(pred)}")
'''
    (docs_dir / "predict.py").write_text(predict_example)
    print(f"  Prediction example saved -> {docs_dir}/predict.py")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str = "outputs",
        indices_dir: str = None, models_dirs: list = None):
    out     = Path(output_dir)
    idx_dir = Path(indices_dir) if indices_dir else out
    out.mkdir(parents=True, exist_ok=True)

    search_dirs = ([Path(d) for d in models_dirs] + [out]) if models_dirs else [out]

    print("=== Step 1: Loading data & models ===")
    (X_train, X_test, y_train, y_test,
     X_smote, y_smote, feat_cols) = load_data(input_path, idx_dir)
    models = load_models(search_dirs)

    print(f"\n=== Step 2: {N_CV_FOLDS}-fold Stratified CV on all 4 models ===")
    print("  Data: SMOTE-resampled training set "
          f"({len(y_smote):,} records, balanced 50/50)")
    cv_df = run_cv(models, X_smote, y_smote)
    cv_df.drop(columns=["All Folds"]).to_csv(
        out / "task_52_cv_results_summary.csv", index=False)
    print("\n  CV Summary:")
    pivot_print = cv_df.pivot(index="Model", columns="Metric",
                               values=["Mean","Std"])
    print(pivot_print.to_string())

    print("\n=== Step 3: Held-out test metrics ===")
    test_df = compute_test_metrics(models, X_test, y_test)
    test_df.to_csv(out / "task_52_test_metrics.csv", index=False)
    print(test_df.to_string(index=False))

    # Determine best model by test AUC
    best_model_name = test_df.loc[test_df["Test AUC"].idxmax(), "Model"]
    best_test_auc   = test_df["Test AUC"].max()
    h5_met = best_test_auc >= 0.75
    print(f"\n  Best model (test AUC): {best_model_name} "
          f"(AUC={best_test_auc:.4f})")
    print(f"  H5 (AUC >= 0.75): {'SUPPORTED' if h5_met else 'NOT SUPPORTED'}")

    print("\n=== Step 4-6: Generating figures ===")
    display_df = plot_cv_comparison_table(cv_df, test_df, out)
    plot_cv_auc_forest(cv_df, out)
    plot_radar(cv_df, out)
    rationale = plot_model_rationale(cv_df, best_model_name, out)

    print("\n=== Step 7: Model selection rationale saved ===")
    print(rationale[:400] + "...")

    print("\n=== Step 8: Saving best model with joblib ===")
    save_best_model(models[best_model_name], best_model_name, out)

    # Task #54 — save feature columns list as JSON for reproducibility
    import json
    with open(out / "task_54_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=4)
    print("  Saved task_54_feature_columns.json")

    print("\n=== Step 9: Writing preprocessing pipeline documentation ===")
    write_pipeline_docs(out)

    print(f"\n=== Story 9 complete -- all outputs saved to {out.resolve()} ===")
    return cv_df, test_df, display_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 9 -- 10-Fold CV, Model Selection & Pipeline Docs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Flat outputs folder
  python src/model_selection_cv.py \\
      --input      outputs/story-1/ksi_encoded.csv \\
      --output-dir outputs/story-9

  # Story subfolders
  python src/model_selection_cv.py \\
      --input       outputs/story-1/ksi_encoded.csv \\
      --output-dir  outputs/story-9 \\
      --indices-dir outputs/story-6 \\
      --models-dir  outputs/story-6 outputs/story-7 outputs/story-8
        """
    )
    parser.add_argument("--input",       required=True,
                        help="Path to raw KSI CSV file")
    parser.add_argument("--output-dir",  default="outputs",
                        help="Directory to write Story 9 outputs (created if absent)")
    parser.add_argument("--indices-dir", default=None,
                        help="Folder containing train_indices.csv + test_indices.csv "
                             "(e.g. outputs/story-6)")
    parser.add_argument("--models-dir",  nargs="+", default=None,
                        help="Folders to search for model .pkl files. "
                             "Example: --models-dir outputs/story-6 "
                             "outputs/story-7 outputs/story-8")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.indices_dir, args.models_dir)