"""
Story 6 – Logistic Regression Baseline ML Classifier
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5

Steps
─────
1.  80/20 stratified train-test split (random_state=42); save indices
2.  Apply SMOTE to training split ONLY
3.  Tune regularisation C via 5-fold stratified CV on SMOTE-resampled train set
4.  Fit final LogisticRegression with best C on full SMOTE train set
5.  Evaluate on imbalanced test set:
      AUC-ROC, precision, recall, F1 (macro + weighted), confusion matrix
6.  Plot ROC curve (Fig 15)
7.  Plot confusion matrix with abs + % labels (Fig 16)
8.  Save model comparison table CSV (baseline row) (Fig 17)
9.  Persist: model, scaler, train/test indices, metrics

Usage:
    python src/ml_logistic_baseline.py \
        --input  outputs/story-1/ksi_encoded.csv \
        --output-dir outputs/story-6
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20

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

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

# ── Palette ───────────────────────────────────────────────────────────────────
C_FATAL    = "#C0392B"
C_NONFATAL = "#2980B9"
C_ACCENT   = "#E67E22"
C_GREEN    = "#27AE60"
C_BG       = "#F8F9FA"
C_GRID_COL = "#DEE2E6"
FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1A1A2E"}
FONT_AX    = {"fontsize": 10, "color": "#2C3E50"}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load & split
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(input_path: str):
    """
    Load ksi_encoded.csv from Story 1 and perform 80/20 stratified split.

    Reads the already-cleaned and encoded dataset — Story 1 has already:
      - Imputed missing values
      - Capped invage outliers (> 110 → NaN → median imputed)
      - OHE-encoded categoricals
      - Encoded target (acclass_binary)
      - Engineered temporal features (hour, is_weekend, etc.)
    """
    df = pd.read_csv(input_path, low_memory=False)

    if "acclass_binary" not in df.columns:
        raise ValueError(
            "Column 'acclass_binary' not found.\n"
            "Run Story 1 first: python src/data_preparation.py --output-dir outputs/story-1\n"
            "Then pass --input outputs/story-1/ksi_encoded.csv"
        )

    cols = [c for c in CORE_FEATURES if c in df.columns]
    missing = [c for c in CORE_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} CORE_FEATURES not found in dataset: {missing[:5]}...")

    X = df[cols].fillna(0)
    y = df["acclass_binary"]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Save indices
    train_idx = X_train.index.tolist()
    test_idx  = X_test.index.tolist()

    print(f"  Total records  : {len(X):,}")
    print(f"  Train size     : {len(X_train):,} ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"  Test size      : {len(X_test):,}  ({TEST_SIZE*100:.0f}%)")
    print(f"  Fatal in train : {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Fatal in test  : {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test, train_idx, test_idx, cols


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — SMOTE on training split only
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train.values, y_train.values)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name="acclass_binary")

    print(f"  Before SMOTE   : {len(y_train):,} samples "
          f"(Fatal={y_train.sum():,}, Non-Fatal={(y_train==0).sum():,})")
    print(f"  After SMOTE    : {len(y_res):,} samples "
          f"(Fatal={y_res.sum():,}, Non-Fatal={(y_res==0).sum():,})")
    print(f"  Test set       : untouched — {0} synthetic samples in test")

    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Tune C with 5-fold stratified CV
# ─────────────────────────────────────────────────────────────────────────────

def tune_regularisation(X_smote: pd.DataFrame, y_smote: pd.Series):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            solver="lbfgs", max_iter=1000,
            class_weight=None,          # SMOTE already balances
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {"clf__C": C_GRID}

    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_smote, y_smote)

    best_C  = grid.best_params_["clf__C"]
    best_auc_cv = grid.best_score_

    print(f"  C values tested : {C_GRID}")
    print(f"  Best C          : {best_C}")
    print(f"  Best CV AUC-ROC : {best_auc_cv:.4f}")

    # Full CV results table
    cv_results = pd.DataFrame(grid.cv_results_)[
        ["param_clf__C", "mean_test_score", "std_test_score", "rank_test_score"]
    ].rename(columns={
        "param_clf__C":    "C",
        "mean_test_score": "Mean AUC (CV)",
        "std_test_score":  "Std AUC (CV)",
        "rank_test_score": "Rank",
    }).sort_values("C")

    return grid.best_estimator_, best_C, best_auc_cv, cv_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    prec_wt,  rec_wt,  f1_wt,  _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    prec_cls, rec_cls, f1_cls, sup = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1], zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "auc":            round(auc_score, 4),
        "precision_macro": round(prec_mac, 4),
        "recall_macro":    round(rec_mac,  4),
        "f1_macro":        round(f1_mac,   4),
        "precision_weighted": round(prec_wt, 4),
        "recall_weighted":    round(rec_wt,  4),
        "f1_weighted":        round(f1_wt,   4),
        "precision_nonfatal": round(prec_cls[0], 4),
        "recall_nonfatal":    round(rec_cls[0],  4),
        "f1_nonfatal":        round(f1_cls[0],   4),
        "precision_fatal":    round(prec_cls[1], 4),
        "recall_fatal":       round(rec_cls[1],  4),
        "f1_fatal":           round(f1_cls[1],   4),
        "cm":    cm,
        "fpr":   fpr,
        "tpr":   tpr,
        "thresholds": thresholds,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel="", ylabel="", title="", grid_axis="both"):
    ax.set_facecolor(C_BG)
    ax.set_title(title, **FONT_TITLE, pad=10)
    ax.set_xlabel(xlabel, **FONT_AX)
    ax.set_ylabel(ylabel, **FONT_AX)
    ax.tick_params(labelsize=9, colors="#4A4A4A")
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID_COL)
    ax.grid(axis=grid_axis, color=C_GRID_COL, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_roc(metrics: dict, best_C: float, best_auc_cv: float, out: Path) -> None:
    fpr, tpr = metrics["fpr"], metrics["tpr"]
    auc_score = metrics["auc"]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")

    # ROC curve
    ax.plot(fpr, tpr, color=C_FATAL, linewidth=2.5,
            label=f"Logistic Regression (AUC = {auc_score:.4f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], color="#95A5A6", linewidth=1.5,
            linestyle="--", label="Random classifier (AUC = 0.50)")

    # AUC ≥ 0.75 threshold line
    ax.axhline(0.75, color=C_ACCENT, linewidth=1.2, linestyle=":",
               label="AUC = 0.75 project target")

    # Shade AUC area
    ax.fill_between(fpr, tpr, alpha=0.08, color=C_FATAL)

    # Annotate AUC on curve
    mid_idx = len(fpr) // 3
    ax.annotate(f"AUC = {auc_score:.4f}",
                xy=(fpr[mid_idx], tpr[mid_idx]),
                xytext=(fpr[mid_idx] + 0.12, tpr[mid_idx] - 0.10),
                fontsize=10, fontweight="bold", color=C_FATAL,
                arrowprops=dict(arrowstyle="->", color=C_FATAL, lw=1.2))

    _style_ax(ax,
              xlabel="False Positive Rate (1 − Specificity)",
              ylabel="True Positive Rate (Sensitivity)",
              title=("Fig 15 — ROC Curve: Logistic Regression Baseline\n"
                     f"Best C = {best_C}  |  5-fold CV AUC = {best_auc_cv:.4f}  |  "
                     f"Test AUC = {auc_score:.4f}"),
              grid_axis="both")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    # Pass / Fail badge
    passed = auc_score >= 0.75
    badge_text = f"{'✓ AUC ≥ 0.75' if passed else '✗ AUC < 0.75'} (target {'met' if passed else 'not met'})"
    badge_color = C_GREEN if passed else C_FATAL
    ax.text(0.97, 0.06, badge_text, transform=ax.transAxes,
            ha="right", fontsize=10, fontweight="bold", color=badge_color,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=badge_color, lw=1.5))

    fig.tight_layout()
    fig.savefig(out / "task_34_roc_curve_logistic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_34_roc_curve_logistic.png")


def plot_confusion_matrix(metrics: dict, y_test: pd.Series, out: Path) -> None:
    cm = metrics["cm"]
    total = cm.sum()
    cm_pct = cm / total * 100

    labels = ["Non-Fatal (0)", "Fatal (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle(
        "Fig 16 — Confusion Matrix: Logistic Regression Baseline\n"
        "Left: absolute counts  |  Right: percentage of all test records",
        fontsize=13, fontweight="bold", color="#1A1A2E"
    )

    for ax, data, fmt_fn, title_sfx in [
        (axes[0], cm,     lambda v: f"{int(v):,}",    "Absolute Counts"),
        (axes[1], cm_pct, lambda v: f"{v:.1f}%",      "Percentage (% of all test records)"),
    ]:
        # Custom heatmap
        vmax = data.max()
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax)

        for i in range(2):
            for j in range(2):
                val = data[i, j]
                text_color = "white" if val > vmax * 0.6 else "#1A1A2E"
                ax.text(j, i, fmt_fn(val),
                        ha="center", va="center",
                        fontsize=16, fontweight="bold", color=text_color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted Label", **FONT_AX)
        ax.set_ylabel("True Label", **FONT_AX)
        ax.set_title(title_sfx, fontsize=11, fontweight="bold", color="#2C3E50", pad=8)

        # Diagonal annotation
        for i, label in enumerate(["True Negative", "True Positive"]):
            ax.text(i, i, f"\n\n{label}", ha="center", va="center",
                    fontsize=8, color="white" if data[i,i] > vmax * 0.6 else "#555",
                    style="italic")

        plt.colorbar(im, ax=ax, shrink=0.85)

    # Metrics annotation panel
    m = metrics
    annot = (
        f"Test set n = {int(y_test.sum()) + int((y_test==0).sum()):,}  "
        f"(Fatal: {int(y_test.sum()):,} | Non-Fatal: {int((y_test==0).sum()):,})\n"
        f"AUC = {m['auc']:.4f}   |   "
        f"Macro F1 = {m['f1_macro']:.4f}   |   "
        f"Weighted F1 = {m['f1_weighted']:.4f}\n"
        f"Fatal recall = {m['recall_fatal']:.4f}   |   "
        f"Fatal precision = {m['precision_fatal']:.4f}"
    )
    fig.text(0.5, -0.03, annot, ha="center", fontsize=9.5,
             color="#2C3E50",
             bbox=dict(boxstyle="round", fc="#EBF5FB", ec="#2980B9", lw=1))

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out / "task_35_confusion_matrix_logistic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_35_confusion_matrix_logistic.png")


def plot_cv_results(cv_results: pd.DataFrame, best_C: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")

    c_vals  = cv_results["C"].astype(float).values
    means   = cv_results["Mean AUC (CV)"].values
    stds    = cv_results["Std AUC (CV)"].values

    colors = [C_FATAL if c == best_C else C_NONFATAL for c in c_vals]
    ax.bar(range(len(c_vals)), means, color=colors,
           edgecolor="white", zorder=3, width=0.6)
    ax.errorbar(range(len(c_vals)), means, yerr=stds,
                fmt="none", color="#1A1A2E", capsize=5,
                linewidth=1.5, capthick=1.5, zorder=4)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.002, f"{m:.4f}", ha="center",
                fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(c_vals)))
    ax.set_xticklabels([f"C={c}" for c in c_vals], fontsize=9)
    ax.set_ylim(min(means) - 0.02, max(means) + 0.025)
    _style_ax(ax,
              xlabel="Regularisation Parameter C",
              ylabel="Mean AUC-ROC (5-fold CV)",
              title=f"Fig 17 — 5-fold CV Tuning of Regularisation Parameter C\nBest C = {best_C} (highlighted in red)",
              grid_axis="y")

    legend_patches = [
        mpatches.Patch(color=C_FATAL,    label=f"Best C = {best_C}"),
        mpatches.Patch(color=C_NONFATAL, label="Other C values"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / "task_36_cv_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_36_cv_tuning.png")


def plot_model_comparison_table(comparison_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 4), facecolor="white")
    fig.suptitle(
        "Fig 18 — Model Comparison Table (Baseline)\n"
        "Evaluated on held-out test set (n = 4,088, imbalanced — no SMOTE)  |  "
        "★ = AUC ≥ 0.75 target met",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")

    col_labels = list(comparison_df.columns)
    cell_data  = comparison_df.values.tolist()

    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.6)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    # Header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1A1A2E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row colours — complete rows green/amber, pending rows grey
    row_fills = {
        "Logistic Regression":  "#D5F5E3",
        "Decision Tree":        "#FEF9E7",
        "Random Forest":        "#FEF9E7",
        "XGBoost":              "#FEF9E7",
    }
    for i, row in enumerate(comparison_df.itertuples(), start=1):
        fill = row_fills.get(row.Model, "white")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fill)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out / "task_37_model_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_37_model_comparison_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str = "outputs"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("═══ Step 1: Load & stratified 80/20 split ═══")
    X_train, X_test, y_train, y_test, \
        train_idx, test_idx, feat_cols = load_and_split(input_path)

    # Save indices
    pd.DataFrame({"train_index": train_idx}).to_csv(
        out / "train_indices.csv", index=False)
    pd.DataFrame({"test_index": test_idx}).to_csv(
        out / "test_indices.csv", index=False)
    print(f"  Train/test indices saved → train_indices.csv, test_indices.csv")

    print("\n═══ Step 2: SMOTE on training split ONLY ═══")
    X_smote, y_smote = apply_smote(X_train, y_train)
    print(f"  ✓ Test set is SMOTE-free — {len(y_test):,} original records only")

    print("\n═══ Step 3: Tune C via 5-fold stratified CV ═══")
    best_model, best_C, best_auc_cv, cv_results = tune_regularisation(X_smote, y_smote)
    cv_results.to_csv(out / "task_36_cv_tuning_results.csv", index=False)
    print(cv_results.to_string(index=False))
    print(f"  CV results saved → cv_tuning_results.csv")

    print(f"\n═══ Step 4: Final model — LogisticRegression (C={best_C}) ═══")
    print(f"  Fitted on SMOTE-resampled train set ({len(y_smote):,} samples)")

    print("\n═══ Step 5: Evaluate on test set ═══")
    metrics = evaluate(best_model, X_test, y_test)

    print(f"\n  ── Metric Summary ──────────────────────────────")
    print(f"  AUC-ROC              : {metrics['auc']:.4f}  "
          f"{'✓ ≥ 0.75 TARGET MET' if metrics['auc'] >= 0.75 else '✗ < 0.75 target not met'}")
    print(f"  Precision (macro)    : {metrics['precision_macro']:.4f}")
    print(f"  Recall    (macro)    : {metrics['recall_macro']:.4f}")
    print(f"  F1        (macro)    : {metrics['f1_macro']:.4f}")
    print(f"  Precision (weighted) : {metrics['precision_weighted']:.4f}")
    print(f"  Recall    (weighted) : {metrics['recall_weighted']:.4f}")
    print(f"  F1        (weighted) : {metrics['f1_weighted']:.4f}")
    print(f"\n  ── Per-class ────────────────────────────────────")
    print(f"  Non-Fatal — P: {metrics['precision_nonfatal']:.4f}  "
          f"R: {metrics['recall_nonfatal']:.4f}  F1: {metrics['f1_nonfatal']:.4f}")
    print(f"  Fatal     — P: {metrics['precision_fatal']:.4f}  "
          f"R: {metrics['recall_fatal']:.4f}  F1: {metrics['f1_fatal']:.4f}")
    print(f"\n  ── Confusion Matrix ──────────────────────────────")
    print(f"  {metrics['cm']}")

    # Save detailed metrics
    metrics_out = {k: v for k, v in metrics.items()
                   if k not in ("cm", "fpr", "tpr", "thresholds", "y_prob", "y_pred")}
    pd.DataFrame([metrics_out]).to_csv(out / "task_34_logistic_baseline_metrics.csv", index=False)
    print(f"\n  Metrics saved → logistic_baseline_metrics.csv")

    # Build model comparison table (baseline row + placeholders)
    target_met = "★" if metrics["auc"] >= 0.75 else ""
    comparison_df = pd.DataFrame([
        {
            "Model":              "Logistic Regression",
            "Best C / Params":    f"C = {best_C}",
            "AUC-ROC":            f"{metrics['auc']:.4f} {target_met}",
            "Precision (macro)":  f"{metrics['precision_macro']:.4f}",
            "Recall (macro)":     f"{metrics['recall_macro']:.4f}",
            "F1 (macro)":         f"{metrics['f1_macro']:.4f}",
            "F1 (weighted)":      f"{metrics['f1_weighted']:.4f}",
            "Fatal Recall":       f"{metrics['recall_fatal']:.4f}",
            "Notes":              "Baseline | SMOTE train | C tuned 5-fold CV",
        },
        {
            "Model":              "Decision Tree",
            "Best C / Params":    "max_depth, min_samples — pending",
            "AUC-ROC":            "—",
            "Precision (macro)":  "—",
            "Recall (macro)":     "—",
            "F1 (macro)":         "—",
            "F1 (weighted)":      "—",
            "Fatal Recall":       "—",
            "Notes":              "Story 7 — pending",
        },
        {
            "Model":              "Random Forest",
            "Best C / Params":    "n_estimators, max_depth — pending",
            "AUC-ROC":            "—",
            "Precision (macro)":  "—",
            "Recall (macro)":     "—",
            "F1 (macro)":         "—",
            "F1 (weighted)":      "—",
            "Fatal Recall":       "—",
            "Notes":              "Story 7 — pending",
        },
        {
            "Model":              "XGBoost",
            "Best C / Params":    "learning_rate, n_estimators — pending",
            "AUC-ROC":            "—",
            "Precision (macro)":  "—",
            "Recall (macro)":     "—",
            "F1 (macro)":         "—",
            "F1 (weighted)":      "—",
            "Fatal Recall":       "—",
            "Notes":              "Story 8 — pending",
        },
    ])
    comparison_df.to_csv(out / "model_comparison_table.csv", index=False)
    print("  Model comparison table saved → model_comparison_table.csv")

    # Persist model
    with open(out / "logistic_baseline_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("  Model persisted → logistic_baseline_model.pkl")

    print("\n═══ Generating figures ═══")
    plot_roc(metrics, best_C, best_auc_cv, out)
    plot_confusion_matrix(metrics, y_test, out)
    plot_cv_results(cv_results, best_C, out)
    plot_model_comparison_table(comparison_df, out)

    print(f"\n═══ Complete — all outputs saved to {out.resolve()} ═══")
    return metrics, best_C, comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 6 – Logistic Regression Baseline ML Classifier")
    parser.add_argument("--input",      required=True,
                        help="Path to ksi_encoded.csv from Story 1 (outputs/story-1/ksi_encoded.csv)")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.input, args.output_dir)