"""
Story 7 – Decision Tree & Random Forest Classifiers
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5

Steps
─────
1.  Reload train/test splits using saved indices (same seed=42 as Story 6)
2.  Apply SMOTE to training split ONLY (same as Story 6)
3.  Decision Tree — GridSearchCV(max_depth 3–15, min_samples_leaf 5–50)
4.  Random Forest — RandomizedSearchCV(n_estimators 100–500,
                                        max_depth 5–20, min_samples_leaf 5–30)
                   Report OOB score
5.  Evaluate both on test set — same metrics as baseline
6.  Append rows to task_41_model_comparison_table.csv
7.  Plot feature importances — top 15 from Random Forest (Fig 19)
8.  Plot learning curves for Random Forest (Fig 20)
9.  Plot combined ROC curves: LR + DT + RF (Fig 21)
10. Plot side-by-side confusion matrices: DT + RF (Fig 22)

Usage:
  # All files in one flat folder (original behaviour)
  python src/ml_decision_tree_rf.py \\
      --input      data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \\
      --output-dir outputs/story-7

  # Story subfolders — indices and LR model in story-6
  python src/ml_decision_tree_rf.py \\
      --input       data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \\
      --output-dir  outputs/story-7 \\
      --indices-dir outputs/story-6 \\
      --models-dir  outputs/story-6
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, learning_curve,
)
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_preparation import (
    encode_categoricals, encode_target, engineer_temporal,
    impute_and_flag, load_dataset,
)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_CV_FOLDS   = 5

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

# ── Palette ───────────────────────────────────────────────────────────────────
C_LR     = "#2980B9"
C_DT     = "#E67E22"
C_RF     = "#27AE60"
C_FATAL  = "#C0392B"
C_BG     = "#F8F9FA"
C_GRID   = "#DEE2E6"
FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1A1A2E"}
FONT_AX    = {"fontsize": 10, "color": "#2C3E50"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(input_path: str, indices_dir: Path):
    """
    Rebuild X/y from raw CSV then slice using saved Story 6 indices.
    indices_dir may differ from output_dir when outputs are in subfolders.
    """
    df = load_dataset(input_path)
    df = impute_and_flag(df)
    df = engineer_temporal(df)
    df = encode_target(df)
    df = encode_categoricals(df)

    cols = [c for c in CORE_FEATURES if c in df.columns]
    X = df[cols].fillna(0)
    y = df["acclass_binary"]

    idx_train = indices_dir / "train_indices.csv"
    idx_test  = indices_dir / "test_indices.csv"

    if not idx_train.exists():
        raise FileNotFoundError(
            f"train_indices.csv not found in: {indices_dir}\n"
            f"Run Story 6 first, or pass the correct folder via --indices-dir."
        )

    train_idx = pd.read_csv(idx_train)["train_index"].tolist()
    test_idx  = pd.read_csv(idx_test)["test_index"].tolist()

    X_train = X.loc[train_idx]
    X_test  = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test  = y.loc[test_idx]

    print(f"  indices_dir   : {indices_dir}")
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"  Fatal in train: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Fatal in test : {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")
    return X_train, X_test, y_train, y_test, cols


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train.values, y_train.values)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name="acclass_binary")
    print(f"  SMOTE: {len(y_train):,} -> {len(y_res):,} "
          f"(Fatal {y_train.sum():,} -> {y_res.sum():,})")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test, name: str) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    pm, rm, fm, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro",    zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    pc, rc, fc, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1], zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    print(f"  AUC-ROC              : {auc_score:.4f}  "
          f"{'✓ >= 0.75 TARGET MET' if auc_score >= 0.75 else '✗ < 0.75 target not met'}")
    print(f"  Precision (macro)    : {pm:.4f}")
    print(f"  Recall    (macro)    : {rm:.4f}")
    print(f"  F1        (macro)    : {fm:.4f}")
    print(f"  F1        (weighted) : {fw:.4f}")
    print(f"  Fatal recall         : {rc[1]:.4f}")
    print(f"  Fatal precision      : {pc[1]:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    return {
        "model_name": name, "auc": auc_score,
        "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "precision_fatal": pc[1], "recall_fatal": rc[1], "f1_fatal": fc[1],
        "precision_nonfatal": pc[0], "recall_nonfatal": rc[0], "f1_nonfatal": fc[0],
        "cm": cm, "fpr": fpr, "tpr": tpr, "y_prob": y_prob, "y_pred": y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Decision Tree
# ─────────────────────────────────────────────────────────────────────────────

def train_decision_tree(X_smote, y_smote, X_test, y_test):
    param_grid = {
        "max_depth":        list(range(3, 16)),
        "min_samples_leaf": [5, 10, 20, 30, 50],
    }
    total_combos = len(param_grid["max_depth"]) * len(param_grid["min_samples_leaf"])
    print(f"\n  GridSearchCV: {total_combos} combos, {N_CV_FOLDS}-fold CV")
    print(f"    max_depth        : {param_grid['max_depth']}")
    print(f"    min_samples_leaf : {param_grid['min_samples_leaf']}")

    cv   = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    dt   = DecisionTreeClassifier(random_state=RANDOM_STATE)
    gscv = GridSearchCV(dt, param_grid, cv=cv, scoring="roc_auc",
                        n_jobs=-1, refit=True, verbose=0)
    gscv.fit(X_smote, y_smote)

    best_params = gscv.best_params_
    best_cv_auc = gscv.best_score_
    best_model  = gscv.best_estimator_

    print(f"\n  Best params : max_depth={best_params['max_depth']}, "
          f"min_samples_leaf={best_params['min_samples_leaf']}")
    print(f"  Best CV AUC : {best_cv_auc:.4f}")

    results_df = pd.DataFrame(gscv.cv_results_)[
        ["param_max_depth", "param_min_samples_leaf",
         "mean_test_score", "std_test_score"]
    ]
    results_df.columns = ["max_depth", "min_samples_leaf", "mean_auc", "std_auc"]

    print("\n  Evaluating on test set:")
    metrics = compute_metrics(best_model, X_test, y_test, "Decision Tree")
    metrics["best_params"]  = best_params
    metrics["best_cv_auc"]  = best_cv_auc
    metrics["grid_results"] = results_df

    return best_model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_smote, y_smote, X_test, y_test, feat_cols):
    param_dist = {
        "n_estimators":     [100, 200, 300, 400, 500],
        "max_depth":        list(range(5, 21)),
        "min_samples_leaf": [5, 10, 15, 20, 30],
        "max_features":     ["sqrt", "log2"],
        "bootstrap":        [True],
    }
    n_iter = 40
    print(f"\n  RandomizedSearchCV: n_iter={n_iter}, {N_CV_FOLDS}-fold CV, scoring=roc_auc")

    cv   = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rf   = RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE, n_jobs=-1)
    rscv = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, cv=cv,
                              scoring="roc_auc", n_jobs=-1,
                              refit=True, verbose=0, random_state=RANDOM_STATE)
    rscv.fit(X_smote, y_smote)

    best_params = rscv.best_params_
    best_cv_auc = rscv.best_score_
    best_model  = rscv.best_estimator_
    oob_score   = best_model.oob_score_

    print(f"\n  Best params:")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<22}: {v}")
    print(f"  Best CV AUC (5-fold): {best_cv_auc:.4f}")
    print(f"  OOB score (accuracy): {oob_score:.4f}")

    print("\n  Evaluating on test set:")
    metrics = compute_metrics(best_model, X_test, y_test, "Random Forest")
    metrics["best_params"] = best_params
    metrics["best_cv_auc"] = best_cv_auc
    metrics["oob_score"]   = oob_score
    metrics["feature_importances"] = pd.Series(
        best_model.feature_importances_, index=feat_cols
    ).sort_values(ascending=False)

    return best_model, metrics


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


def plot_dt_grid_heatmap(grid_results: pd.DataFrame,
                         best_params: dict, out: Path) -> None:
    pivot = grid_results.pivot(index="max_depth",
                               columns="min_samples_leaf",
                               values="mean_auc")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    fig.suptitle(
        "Fig 19a -- Decision Tree GridSearchCV Heatmap\n"
        "Mean 5-fold CV AUC-ROC across max_depth x min_samples_leaf",
        **FONT_TITLE, y=0.98
    )
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=pivot.values.min(), vmax=pivot.values.max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticklabels(pivot.index,   fontsize=9)
    ax.set_xlabel("min_samples_leaf", **FONT_AX)
    ax.set_ylabel("max_depth",        **FONT_AX)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if val > pivot.values.mean() + 0.003 else "#1A1A2E")

    best_d   = list(pivot.index).index(best_params["max_depth"])
    best_msl = list(pivot.columns).index(best_params["min_samples_leaf"])
    ax.add_patch(plt.Rectangle((best_msl - 0.5, best_d - 0.5), 1, 1,
                                fill=False, edgecolor="#27AE60",
                                linewidth=3, zorder=5))
    ax.text(best_msl, best_d - 0.38,
            f"BEST: depth={best_params['max_depth']}, leaf={best_params['min_samples_leaf']}",
            ha="center", va="top", fontsize=7.5,
            color="#27AE60", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean CV AUC-ROC", shrink=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "task_39_dt_grid_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_39_dt_grid_heatmap.png")


def plot_feature_importances(importances: pd.Series, out: Path) -> None:
    top15  = importances.head(15).sort_values(ascending=True)
    labels = [n.replace("impactype_","Impact: ").replace("traffictl_","Control: ")
               .replace("road_class_","Road: ").replace("accloc_","Loc: ")
               .replace("rdsfcond_","Surface: ").replace("light_","Light: ")
               .replace("_"," ") for n in top15.index]
    values = top15.values
    colors = ["#C0392B" if v >= values.mean() else "#2980B9" for v in values]

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    ax.barh(range(len(labels)), values, color=colors, edgecolor="white", zorder=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)

    for i, v in enumerate(values):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=8.5)

    ax.axvline(values.mean(), color="#E67E22", linestyle="--",
               linewidth=1.5)
    _style_ax(ax,
              xlabel="Mean Decrease in Impurity (Gini Importance)",
              title="Fig 19b -- Random Forest: Top 15 Feature Importances\n"
                    "(higher = more predictive of fatal collision outcome)",
              grid_axis="x")

    ax.legend(handles=[
        mpatches.Patch(color="#C0392B", label="Above-average importance"),
        mpatches.Patch(color="#2980B9", label="Below-average importance"),
        plt.Line2D([0],[0], color="#E67E22", linestyle="--",
                   lw=1.5, label=f"Mean ({values.mean():.4f})"),
    ], fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / "task_42_rf_feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_42_rf_feature_importances.png")

    importances.reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    ).to_csv(out / "task_42_rf_feature_importances.csv", index=False)


def plot_learning_curves(rf_model, X_smote, y_smote, out: Path) -> None:
    print("  Computing learning curves (may take ~60s)...")
    train_sizes = np.linspace(0.05, 1.0, 12)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sz, train_sc, val_sc = learning_curve(
        rf_model, X_smote, y_smote,
        train_sizes=train_sizes, cv=cv, scoring="roc_auc",
        n_jobs=-1, shuffle=True, random_state=RANDOM_STATE,
    )
    train_mean = train_sc.mean(axis=1); train_std = train_sc.std(axis=1)
    val_mean   = val_sc.mean(axis=1);   val_std   = val_sc.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.plot(train_sz, train_mean, "o-", color=C_RF, linewidth=2.2,
            markersize=5, label="Training AUC-ROC")
    ax.fill_between(train_sz, train_mean - train_std,
                    train_mean + train_std, alpha=0.12, color=C_RF)
    ax.plot(train_sz, val_mean, "s--", color=C_DT, linewidth=2.2,
            markersize=5, label="5-fold CV AUC-ROC")
    ax.fill_between(train_sz, val_mean - val_std,
                    val_mean + val_std, alpha=0.12, color=C_DT)
    ax.axhline(0.75, color="#C0392B", linewidth=1.3, linestyle=":",
               label="AUC = 0.75 project target")

    gap = train_mean[-1] - val_mean[-1]
    diag, diag_col = (
        ("Converging curves -> low bias, low variance (good fit)", "#27AE60")
        if gap < 0.03 else
        ("Mild gap -> slight overfit; more data or regularisation may help", "#E67E22")
        if gap < 0.08 else
        ("Large gap -> overfitting; reduce depth or increase regularisation", "#C0392B")
    )
    ax.text(0.03, 0.05, f"Gap={gap:.4f}: {diag}",
            transform=ax.transAxes, fontsize=9, color=diag_col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=diag_col, lw=1.3))

    _style_ax(ax,
              xlabel="Training Set Size (records)", ylabel="AUC-ROC Score",
              title="Fig 20 -- Random Forest Learning Curves\n"
                    "Diagnosis: training vs. CV score convergence as data grows",
              grid_axis="both")
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=9.5, framealpha=0.95, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "task_43_rf_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_43_rf_learning_curves.png")


def plot_combined_roc(lr_metrics: dict, dt_metrics: dict,
                      rf_metrics: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    for m, label, color, lw, ls in [
        (lr_metrics, f"Logistic Regression (AUC={lr_metrics['auc']:.4f})", C_LR, 1.8, "--"),
        (dt_metrics, f"Decision Tree       (AUC={dt_metrics['auc']:.4f})", C_DT, 2.0, "-."),
        (rf_metrics, f"Random Forest       (AUC={rf_metrics['auc']:.4f})", C_RF, 2.5, "-"),
    ]:
        ax.plot(m["fpr"], m["tpr"], color=color, linewidth=lw, linestyle=ls, label=label)

    ax.plot([0,1],[0,1], color="#95A5A6", linewidth=1.3, linestyle=":",
            label="Random classifier (AUC=0.50)")
    ax.axhline(0.75, color="#C0392B", linewidth=1.0, linestyle=":", alpha=0.6,
               label="AUC=0.75 target")
    ax.fill_between(rf_metrics["fpr"], rf_metrics["tpr"], alpha=0.06, color=C_RF)

    _style_ax(ax,
              xlabel="False Positive Rate (1 - Specificity)",
              ylabel="True Positive Rate (Sensitivity)",
              title="Fig 21 -- ROC Curves: All Models Compared\n"
                    "Evaluated on held-out test set (n=4,088, imbalanced, no SMOTE)",
              grid_axis="both")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
              prop={"family": "monospace"})

    best_auc  = max(lr_metrics["auc"], dt_metrics["auc"], rf_metrics["auc"])
    best_name = {lr_metrics["auc"]:"Logistic Regression",
                 dt_metrics["auc"]:"Decision Tree",
                 rf_metrics["auc"]:"Random Forest"}[best_auc]
    badge_col  = "#27AE60" if best_auc >= 0.75 else "#E67E22"
    badge_text = f"Best: {best_name}\nAUC = {best_auc:.4f}" + \
                 (" ✓" if best_auc >= 0.75 else " ✗")
    ax.text(0.97, 0.10, badge_text, transform=ax.transAxes,
            ha="right", fontsize=9.5, fontweight="bold", color=badge_col,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=badge_col, lw=1.5))

    fig.tight_layout()
    fig.savefig(out / "task_41_combined_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_41_combined_roc.png")


def plot_side_by_side_cm(dt_metrics: dict, rf_metrics: dict,
                          y_test, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="white")
    fig.suptitle(
        "Fig 22 -- Confusion Matrices: Decision Tree & Random Forest\n"
        "Left: absolute counts  |  Right: % of all test records",
        fontsize=13, fontweight="bold", color="#1A1A2E"
    )
    labels = ["Non-Fatal (0)", "Fatal (1)"]
    total  = len(y_test)

    for row_idx, (m, model_name) in enumerate([
        (dt_metrics, "Decision Tree"),
        (rf_metrics, "Random Forest"),
    ]):
        cm     = m["cm"]
        cm_pct = cm / total * 100

        for col_idx, (data, fmt_fn, col_title) in enumerate([
            (cm,     lambda v: f"{int(v):,}", "Absolute Counts"),
            (cm_pct, lambda v: f"{v:.1f}%",  "Percentage"),
        ]):
            ax   = axes[row_idx][col_idx]
            vmax = data.max()
            im   = ax.imshow(data,
                             cmap="Greens" if model_name == "Random Forest" else "Oranges",
                             vmin=0, vmax=vmax)
            for i in range(2):
                for j in range(2):
                    v = data[i, j]
                    ax.text(j, i, fmt_fn(v), ha="center", va="center",
                            fontsize=15, fontweight="bold",
                            color="white" if v > vmax * 0.55 else "#1A1A2E")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True",      fontsize=9)
            ax.set_title(f"{model_name} -- {col_title}\n"
                         f"AUC={m['auc']:.4f}  Fatal Recall={m['recall_fatal']:.4f}",
                         fontsize=10, fontweight="bold", color="#2C3E50")
            plt.colorbar(im, ax=ax, shrink=0.82)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "task_41_confusion_matrices_dt_rf.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_41_confusion_matrices_dt_rf.png")


def plot_comparison_table_fig(comparison_df: pd.DataFrame, out: Path) -> None:
    cols    = ["Model","Best Params","AUC-ROC","Precision\n(macro)",
               "Recall\n(macro)","F1\n(macro)","F1\n(weighted)","Fatal\nRecall","Notes"]
    display = comparison_df[cols].copy()

    fig, ax = plt.subplots(figsize=(20, 4.5), facecolor="white")
    fig.suptitle(
        "Fig 23 -- Model Comparison Table (Stories 6 & 7)\n"
        "Evaluated on held-out test set (n=4,088)  |  * = AUC >= 0.75 target met",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")
    tbl = ax.table(cellText=display.values.tolist(), colLabels=cols,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.8)
    tbl.auto_set_column_width(col=list(range(len(cols))))

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#1A1A2E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fills = ["#D5F5E3","#FEF9E7","#D5F5E3","#F4F6F7"]
    for i, fill in enumerate(fills, start=1):
        if i <= len(display):
            for j in range(len(cols)):
                tbl[i, j].set_facecolor(fill)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out / "task_41_model_comparison_table.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_41_model_comparison_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str = "outputs",
        indices_dir: str = None, models_dirs: list = None):
    out     = Path(output_dir)
    idx_dir = Path(indices_dir) if indices_dir else out
    out.mkdir(parents=True, exist_ok=True)

    # Directories to search for prior model pkl files (e.g. logistic_baseline_model.pkl)
    model_search_dirs = [Path(d) for d in models_dirs] + [out] if models_dirs else [out]

    def find_model(filename: str) -> Path:
        for d in model_search_dirs:
            p = d / filename
            if p.exists():
                return p
        searched = ", ".join(str(d) for d in model_search_dirs)
        raise FileNotFoundError(
            f"{filename} not found in: {searched}\n"
            f"Run Story 6 first, or pass the correct folder via --models-dir."
        )

    def find_comparison_table() -> pd.DataFrame:
        for d in model_search_dirs + [out]:
            p = d / "task_41_model_comparison_table.csv"
            if p.exists():
                print(f"  Loaded comparison table from: {p}")
                return pd.read_csv(p)
        return pd.DataFrame()

    print("=== Loading data & reusing Story 6 train/test splits ===")
    X_train, X_test, y_train, y_test, feat_cols = load_splits(input_path, idx_dir)

    print("\n=== Applying SMOTE to training split ONLY ===")
    X_smote, y_smote = apply_smote(X_train, y_train)

    # Load LR baseline for combined ROC
    try:
        with open(find_model("logistic_baseline_model.pkl"), "rb") as f:
            lr_model = pickle.load(f)
        print("\n  LR baseline model loaded OK")
        lr_metrics = compute_metrics(lr_model, X_test, y_test, "Logistic Regression")
    except FileNotFoundError as e:
        lr_metrics = None
        print(f"\n  WARNING: {e}\n  Combined ROC will be skipped.")

    # Decision Tree
    print("\n=== Step 3: Decision Tree -- GridSearchCV ===")
    dt_model, dt_metrics = train_decision_tree(X_smote, y_smote, X_test, y_test)
    with open(out / "dt_model.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    print("  Model saved -> dt_model.pkl")

    # Random Forest
    print("\n=== Step 4: Random Forest -- RandomizedSearchCV ===")
    rf_model, rf_metrics = train_random_forest(
        X_smote, y_smote, X_test, y_test, feat_cols)
    with open(out / "rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("  Model saved -> rf_model.pkl")

    # Build comparison table
    print("\n=== Updating model comparison table ===")

    def fmt_params(p):
        return ", ".join(f"{k.replace('_',' ')}={v}"
                         for k, v in sorted(p.items()) if k != "bootstrap")

    def make_row(name, m, params_str, notes):
        t = "*" if m["auc"] >= 0.75 else ""
        return {"Model": name, "Best Params": params_str,
                "AUC-ROC": f"{m['auc']:.4f} {t}",
                "Precision\n(macro)": f"{m['precision_macro']:.4f}",
                "Recall\n(macro)":    f"{m['recall_macro']:.4f}",
                "F1\n(macro)":        f"{m['f1_macro']:.4f}",
                "F1\n(weighted)":     f"{m['f1_weighted']:.4f}",
                "Fatal\nRecall":      f"{m['recall_fatal']:.4f}",
                "Notes": notes}

    existing = find_comparison_table()

    if lr_metrics:
        lr_row = make_row("Logistic Regression", lr_metrics,
                          "C=0.01 (5-fold CV tuned)", "Baseline -- Story 6")
    elif len(existing) and "Logistic Regression" in existing["Model"].values:
        lr_row = existing[existing["Model"] == "Logistic Regression"].iloc[0].to_dict()
    else:
        lr_row = {"Model": "Logistic Regression", "Best Params": "C=0.01",
                  "AUC-ROC": "0.6679",
                  **{k: "--" for k in ["Precision\n(macro)","Recall\n(macro)",
                                       "F1\n(macro)","F1\n(weighted)","Fatal\nRecall"]},
                  "Notes": "Baseline -- Story 6"}

    comparison_df = pd.DataFrame([
        lr_row,
        make_row("Decision Tree", dt_metrics, fmt_params(dt_metrics["best_params"]),
                 f"GridSearchCV 5-fold | CV AUC={dt_metrics['best_cv_auc']:.4f}"),
        make_row("Random Forest", rf_metrics, fmt_params(rf_metrics["best_params"]),
                 f"RandomizedSearchCV 5-fold | OOB={rf_metrics['oob_score']:.4f}"),
        {"Model": "XGBoost", "Best Params": "pending", "AUC-ROC": "--",
         **{k: "--" for k in ["Precision\n(macro)","Recall\n(macro)",
                               "F1\n(macro)","F1\n(weighted)","Fatal\nRecall"]},
         "Notes": "Story 8 -- pending"},
    ])

    comparison_df.to_csv(out / "task_41_model_comparison_table.csv", index=False)
    print("  model_comparison_table.csv saved")
    print(comparison_df[["Model","AUC-ROC","F1\n(macro)",
                          "Fatal\nRecall","Notes"]].to_string(index=False))

    # Save per-model metric CSVs
    for name, m in [("dt", dt_metrics), ("rf", rf_metrics)]:
        safe = {k: v for k, v in m.items()
                if k not in ("cm","fpr","tpr","y_prob","y_pred",
                             "feature_importances","grid_results","best_params")}
        safe["best_params"] = str(m.get("best_params", {}))
        pd.DataFrame([safe]).to_csv(out / f"task_41_{name}_metrics.csv", index=False)

    # Figures
    print("\n=== Generating figures ===")
    plot_dt_grid_heatmap(dt_metrics["grid_results"], dt_metrics["best_params"], out)
    plot_feature_importances(rf_metrics["feature_importances"], out)
    plot_learning_curves(rf_model, X_smote, y_smote, out)

    if lr_metrics:
        plot_combined_roc(lr_metrics, dt_metrics, rf_metrics, out)
    else:
        print("  Skipping combined ROC (LR model not found)")

    plot_side_by_side_cm(dt_metrics, rf_metrics, y_test, out)
    plot_comparison_table_fig(comparison_df, out)

    print(f"\n=== Story 7 complete -- all outputs saved to {out.resolve()} ===")
    return dt_metrics, rf_metrics, comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 7 -- Decision Tree & Random Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All files in one flat folder
  python src/ml_decision_tree_rf.py \\
      --input      data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \\
      --output-dir outputs/story-7

  # Story subfolders -- indices and LR model in story-6
  python src/ml_decision_tree_rf.py \\
      --input       data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \\
      --output-dir  outputs/story-7 \\
      --indices-dir outputs/story-6 \\
      --models-dir  outputs/story-6
        """
    )
    parser.add_argument("--input",       required=True,
                        help="Path to raw KSI CSV file")
    parser.add_argument("--output-dir",  default="outputs",
                        help="Directory to write Story 7 outputs (created if absent)")
    parser.add_argument("--indices-dir", default=None,
                        help="Folder containing train_indices.csv and test_indices.csv "
                             "(defaults to --output-dir if not set)")
    parser.add_argument("--models-dir",  nargs="+", default=None,
                        help="One or more folders to search for logistic_baseline_model.pkl. "
                             "Example: --models-dir outputs/story-6")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.indices_dir, args.models_dir)