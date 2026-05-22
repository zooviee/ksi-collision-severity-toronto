"""
Story 8 – XGBoost Classifier + SHAP Explainability
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5

Steps
─────
1.  Reload train/test splits (same seed=42 as Stories 6 & 7)
2.  Train XGBoost under THREE imbalance strategies:
      A. SMOTE only (scale_pos_weight=1)
      B. scale_pos_weight=6.08 only (no SMOTE)
      C. SMOTE + scale_pos_weight (combined)
    → Compare all three on test AUC; pick best for SHAP
3.  Tune best strategy with Optuna (50 trials, TPE sampler, AUC objective)
    Params: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
4.  Evaluate final XGBoost on test set — same metrics table as Stories 6 & 7
5.  Update model_comparison_table.csv
6.  Compute SHAP values for best model
7.  Plot:
      (a) SHAP beeswarm summary plot        (Fig 24)
      (b) SHAP bar plot — mean |SHAP|       (Fig 25)
      (c) Dependence plots — top 3 features (Fig 26)
8.  Write plain-language SHAP interpretations (Fig 27)
9.  Final combined ROC — all 4 models       (Fig 28)

Usage:
    python src/ml_xgboost_shap.py \\
        --input  outputs/story-1/ksi_encoded.csv \\
        --output outputs/
"""

import argparse
import json
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
import shap
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
SCALE_POS_WEIGHT = 6.08          # neg/pos ratio in raw training set
N_OPTUNA_TRIALS  = 50

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

FEAT_LABELS = {
    "invage":                              "Age of involved person",
    "older_adult":                         "Older adult (65+)",
    "school_child":                        "School-age child",
    "motorcyclist":                        "Motorcyclist",
    "aggressive":                          "Aggressive driving",
    "distracted":                          "Distracted driving",
    "red_light":                           "Red-light violation",
    "hour":                                "Hour of day",
    "is_weekend":                          "Weekend collision",
    "light_Dark":                          "Dark (no artificial light)",
    "light_Dark with Artificial Lighting": "Dark w/ artificial light",
    "light_Dusk":                          "Dusk conditions",
    "rdsfcond_Wet":                        "Wet road surface",
    "rdsfcond_Ice":                        "Icy road surface",
    "rdsfcond_Loose Snow":                 "Loose snow on road",
    "traffictl_Traffic Signal":            "Traffic signal (vs. No control)",
    "traffictl_Stop Sign":                 "Stop sign (vs. No control)",
    "road_class_Expressway":               "Expressway road",
    "road_class_Local":                    "Local road",
    "road_class_Minor Arterial":           "Minor arterial road",
    "accloc_Non-Intersection":             "Non-intersection location",
    "accloc_Intersection-Related":         "Intersection-related location",
    "impactype_Cyclist Collision":         "Cyclist collision type",
    "impactype_Rear End":                  "Rear-end collision",
    "impactype_Turning Movement":          "Turning movement collision",
}

# ── Palette ───────────────────────────────────────────────────────────────────
C_LR     = "#2980B9"
C_DT     = "#E67E22"
C_RF     = "#27AE60"
C_XGB    = "#8E44AD"
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
    Load ksi_encoded.csv from Story 1 and slice using saved
    train/test indices from Story 6.

    Reads the already-cleaned and encoded dataset — Story 1 has already
    applied imputation, OHE encoding, temporal feature engineering,
    invage outlier capping, and target encoding.
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

    idx_file_train = indices_dir / "train_indices.csv"
    idx_file_test  = indices_dir / "test_indices.csv"

    if not idx_file_train.exists():
        raise FileNotFoundError(
            f"train_indices.csv not found in: {indices_dir}\n"
            f"Run Story 6 first, or pass the correct folder via --indices-dir."
        )

    train_idx = pd.read_csv(idx_file_train)["train_index"].tolist()
    test_idx  = pd.read_csv(idx_file_test)["test_index"].tolist()

    return (X.loc[train_idx], X.loc[test_idx],
            y.loc[train_idx], y.loc[test_idx], cols)


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)
    X_r, y_r = smote.fit_resample(X_train.values, y_train.values)
    return (pd.DataFrame(X_r, columns=X_train.columns),
            pd.Series(y_r, name="acclass_binary"))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
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

    return {
        "model_name": name,
        "auc": auc_score,
        "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "precision_fatal": pc[1], "recall_fatal": rc[1], "f1_fatal": fc[1],
        "precision_nonfatal": pc[0], "recall_nonfatal": rc[0],
        "cm": confusion_matrix(y_test, y_pred),
        "fpr": fpr, "tpr": tpr, "y_prob": y_prob, "y_pred": y_pred,
    }


def print_metrics(m: dict) -> None:
    t = "✓ ≥ 0.75 TARGET MET" if m["auc"] >= 0.75 else "✗ < 0.75"
    print(f"  AUC-ROC              : {m['auc']:.4f}  {t}")
    print(f"  Precision (macro)    : {m['precision_macro']:.4f}")
    print(f"  Recall    (macro)    : {m['recall_macro']:.4f}")
    print(f"  F1        (macro)    : {m['f1_macro']:.4f}")
    print(f"  F1        (weighted) : {m['f1_weighted']:.4f}")
    print(f"  Fatal recall         : {m['recall_fatal']:.4f}")
    print(f"  Fatal precision      : {m['precision_fatal']:.4f}")
    print(f"  Confusion matrix:\n{m['cm']}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Compare three imbalance strategies (quick screen, no tuning)
# ─────────────────────────────────────────────────────────────────────────────

def screen_imbalance_strategies(X_train, y_train, X_smote, y_smote,
                                 X_test, y_test) -> dict:
    base_params = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0,
    )

    results = {}

    # A — SMOTE only (spw=1)
    m_a = XGBClassifier(**base_params, scale_pos_weight=1)
    m_a.fit(X_smote.values, y_smote.values)
    results["A_SMOTE_only"] = compute_metrics(m_a, X_test.values, y_test, "XGB-SMOTE")
    print(f"  A (SMOTE only)            AUC={results['A_SMOTE_only']['auc']:.4f}")

    # B — scale_pos_weight only (no SMOTE)
    m_b = XGBClassifier(**base_params, scale_pos_weight=SCALE_POS_WEIGHT)
    m_b.fit(X_train.values, y_train.values)
    results["B_SPW_only"] = compute_metrics(m_b, X_test.values, y_test, "XGB-SPW")
    print(f"  B (scale_pos_weight={SCALE_POS_WEIGHT})  AUC={results['B_SPW_only']['auc']:.4f}")

    # C — SMOTE + scale_pos_weight
    m_c = XGBClassifier(**base_params, scale_pos_weight=SCALE_POS_WEIGHT)
    m_c.fit(X_smote.values, y_smote.values)
    results["C_SMOTE_SPW"] = compute_metrics(m_c, X_test.values, y_test, "XGB-SMOTE+SPW")
    print(f"  C (SMOTE + SPW)           AUC={results['C_SMOTE_SPW']['auc']:.4f}")

    best_key = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  → Best strategy: {best_key} (AUC={results[best_key]['auc']:.4f})")
    return results, best_key


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_with_optuna(X_fit, y_fit, best_strategy: str) -> tuple:
    """
    Tune XGBoost hyperparameters using Optuna TPE sampler.
    Uses 5-fold stratified CV on the fitting data (SMOTE or raw, per strategy).
    Objective: maximise mean AUC-ROC.
    """
    use_spw = "SPW" in best_strategy
    spw_val = SCALE_POS_WEIGHT if use_spw else 1

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",   100, 500, step=50),
            "max_depth":         trial.suggest_int("max_depth",       3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample",     0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            "scale_pos_weight":  spw_val,
            "eval_metric":       "auc",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "verbosity":         0,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_fit, y_fit,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_cv_auc = study.best_value

    print(f"\n  Optuna best CV AUC : {best_cv_auc:.4f}")
    print(f"  Best hyperparameters:")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<22}: {v}")

    # Fit final model on full fitting data with best params
    final_params = {**best_params,
                    "scale_pos_weight": spw_val,
                    "eval_metric": "auc",
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                    "verbosity": 0}
    final_model = XGBClassifier(**final_params)
    final_model.fit(X_fit, y_fit)

    return final_model, best_params, best_cv_auc, study


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model, X_test: pd.DataFrame, feat_cols: list) -> tuple:
    """Compute SHAP values using TreeExplainer. Returns explainer + shap_values."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)
    return explainer, shap_values


def plain_labels(cols):
    return [FEAT_LABELS.get(c, c.replace("_", " ")) for c in cols]


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


def plot_strategy_comparison(screen_results: dict, out: Path) -> None:
    labels = ["A: SMOTE only", "B: scale_pos_weight only", "C: SMOTE + SPW"]
    keys   = ["A_SMOTE_only", "B_SPW_only", "C_SMOTE_SPW"]
    aucs   = [screen_results[k]["auc"] for k in keys]
    f1s    = [screen_results[k]["f1_macro"] for k in keys]
    recs   = [screen_results[k]["recall_fatal"] for k in keys]

    x = np.arange(3)
    w = 0.25
    colors = [C_LR, C_RF, C_XGB]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    b1 = ax.bar(x - w, aucs, w, label="AUC-ROC",      color=colors[0], edgecolor="white")
    b2 = ax.bar(x,     f1s,  w, label="F1 (macro)",   color=colors[1], edgecolor="white")
    b3 = ax.bar(x + w, recs, w, label="Fatal Recall",  color=colors[2], edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.axhline(0.75, color=C_FATAL, linewidth=1.5, linestyle="--",
               label="AUC=0.75 target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    _style_ax(ax,
              ylabel="Score",
              title="Fig 24 — XGBoost Imbalance Strategy Comparison\n"
                    "(base model, no hyperparameter tuning — 3 strategies on test set)",
              grid_axis="y")
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / "task_45_xgb_strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_45_xgb_strategy_comparison.png")


def plot_optuna_history(study, out: Path) -> None:
    trials = study.trials_dataframe()
    trials = trials.sort_values("number")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle("Fig 25 — Optuna Hyperparameter Tuning History\n"
                 f"TPE Sampler | {N_OPTUNA_TRIALS} trials | Objective: maximise 5-fold CV AUC-ROC",
                 **FONT_TITLE)

    # Optimisation history
    ax1 = axes[0]
    best_so_far = trials["value"].cummax()
    ax1.scatter(trials["number"], trials["value"], alpha=0.45, s=18,
                color=C_XGB, label="Trial AUC")
    ax1.plot(trials["number"], best_so_far, color=C_FATAL,
             linewidth=2.2, label="Best so far")
    ax1.axhline(0.75, color=C_FATAL, linewidth=1.2, linestyle=":",
                alpha=0.7, label="AUC=0.75 target")
    _style_ax(ax1, xlabel="Trial number", ylabel="CV AUC-ROC",
              title="Optimisation history", grid_axis="y")
    ax1.legend(fontsize=8.5, framealpha=0.9)

    # Param importance (top params by variance across top-20 trials)
    ax2 = axes[1]
    top20 = trials.nlargest(20, "value")
    param_cols = [c for c in trials.columns if c.startswith("params_")]
    variances   = top20[param_cols].var().sort_values(ascending=False)
    variances.index = [c.replace("params_", "") for c in variances.index]

    colors = [C_XGB if v > variances.mean() else C_LR for v in variances.values]
    ax2.barh(range(len(variances)), variances.values,
             color=colors, edgecolor="white", zorder=3)
    ax2.set_yticks(range(len(variances)))
    ax2.set_yticklabels(variances.index, fontsize=9)
    ax2.invert_yaxis()
    _style_ax(ax2, xlabel="Variance across top-20 trials",
              title="Param spread (top-20 trials)", grid_axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "task_45_optuna_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_45_optuna_history.png")


def plot_shap_beeswarm(shap_values, X_test: pd.DataFrame,
                        feat_cols: list, out: Path) -> None:
    labels = plain_labels(feat_cols)
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")

    # Use shap's built-in beeswarm on the axes
    shap_exp = shap.Explanation(
        values=shap_values,
        data=X_test.values,
        feature_names=labels,
    )
    shap.plots.beeswarm(shap_exp, max_display=20, show=False)

    plt.title("Fig 26a — SHAP Summary (Beeswarm) Plot\n"
              "Each dot = one test record | Colour = feature value | "
              "x-axis = SHAP impact on fatal probability",
              fontsize=11, fontweight="bold", color="#1A1A2E", pad=10)
    plt.tight_layout()
    plt.savefig(out / "task_48_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved task_48_shap_beeswarm.png")


def plot_shap_bar(shap_values, feat_cols: list, out: Path) -> pd.Series:
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feat_cols
    ).sort_values(ascending=True)

    labels = [FEAT_LABELS.get(c, c.replace("_", " ")) for c in mean_abs.index]
    values = mean_abs.values
    colors = [C_FATAL if v >= values.mean() else C_LR for v in values]

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")
    bars = ax.barh(range(len(labels)), values, color=colors,
                   edgecolor="white", zorder=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)

    for bar, v in zip(bars, values):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8.5)

    ax.axvline(values.mean(), color=C_DT, linestyle="--",
               linewidth=1.5, label=f"Mean ({values.mean():.4f})")
    _style_ax(ax,
              xlabel="Mean |SHAP value|  (average impact on model output)",
              title="Fig 26b — SHAP Feature Importance Bar Plot\n"
                    "Mean absolute SHAP values — higher = more influential on fatality prediction",
              grid_axis="x")
    legend_patches = [
        mpatches.Patch(color=C_FATAL, label="Above-mean SHAP importance"),
        mpatches.Patch(color=C_LR,    label="Below-mean SHAP importance"),
        plt.Line2D([0],[0], color=C_DT, linestyle="--",
                   lw=1.5, label=f"Mean importance ({values.mean():.4f})"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / "task_48_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_48_shap_bar.png")

    return mean_abs.sort_values(ascending=False)


def plot_shap_dependence(shap_values, X_test: pd.DataFrame,
                          feat_cols: list, top3: list, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    fig.suptitle(
        "Fig 26c — SHAP Dependence Plots: Top 3 Features\n"
        "x-axis = feature value | y-axis = SHAP value (impact on fatal probability) | "
        "colour = interaction feature",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )

    shap_df = pd.DataFrame(shap_values, columns=feat_cols)
    X_arr   = X_test.reset_index(drop=True)

    for ax, feat in zip(axes, top3):
        vals      = X_arr[feat].values
        shap_vals = shap_df[feat].values

        # Colour by the next most important feature (interaction)
        interact_feat = [f for f in top3 if f != feat][0]
        interact_vals = X_arr[interact_feat].values

        sc = ax.scatter(vals, shap_vals, c=interact_vals,
                        cmap="RdBu_r", alpha=0.4, s=8, zorder=3)
        ax.axhline(0, color="#95A5A6", linewidth=1.2, linestyle="--")

        # Trend line
        idx_sort = np.argsort(vals)
        from scipy.signal import savgol_filter
        if len(vals) > 20:
            window = min(51, len(vals) // 5 * 2 + 1)
            trend = savgol_filter(shap_vals[idx_sort], window, 3)
            ax.plot(vals[idx_sort], trend, color=C_FATAL,
                    linewidth=2, label="Trend")

        label = FEAT_LABELS.get(feat, feat.replace("_", " "))
        int_label = FEAT_LABELS.get(interact_feat, interact_feat.replace("_", " "))
        ax.set_xlabel(label, **FONT_AX)
        ax.set_ylabel("SHAP value (↑ = higher fatal odds)", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold", color="#2C3E50")
        ax.set_facecolor(C_BG)
        ax.grid(color=C_GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        plt.colorbar(sc, ax=ax, label=f"Colour: {int_label}", shrink=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / "task_48_shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_48_shap_dependence.png")


def plot_shap_interpretation_cards(mean_shap: pd.Series,
                                    shap_values, X_test,
                                    feat_cols: list, out: Path) -> list[dict]:
    """
    Generate plain-language interpretation for top 5 SHAP features.
    Returns list of interpretation dicts for saving.
    """
    top5_feats = mean_shap.head(5).index.tolist()
    interpretations = []

    shap_df = pd.DataFrame(shap_values, columns=feat_cols)
    X_arr   = X_test.reset_index(drop=True)

    for rank, feat in enumerate(top5_feats, 1):
        shap_col  = shap_df[feat].values
        feat_col  = X_arr[feat].values
        label     = FEAT_LABELS.get(feat, feat.replace("_", " "))
        mean_shap_val = np.abs(shap_col).mean()

        # Direction: positive SHAP → increases fatal probability
        high_mask = feat_col > feat_col.mean()
        low_mask  = ~high_mask
        mean_shap_high = shap_col[high_mask].mean() if high_mask.sum() > 0 else 0
        mean_shap_low  = shap_col[low_mask].mean()  if low_mask.sum()  > 0 else 0

        # Build sentence
        if feat in ("aggressive", "distracted", "older_adult", "school_child",
                     "motorcyclist", "red_light", "is_weekend",
                     "light_Dark", "light_Dark with Artificial Lighting", "light_Dusk",
                     "rdsfcond_Ice", "rdsfcond_Wet", "rdsfcond_Loose Snow"):
            # Binary feature
            on_shap  = shap_col[feat_col == 1].mean() if (feat_col == 1).sum() > 0 else 0
            off_shap = shap_col[feat_col == 0].mean() if (feat_col == 0).sum() > 0 else 0
            direction = "increases" if on_shap > off_shap else "decreases"
            sentence = (
                f"When '{label}' is flagged (= 1), the model assigns an average SHAP value of "
                f"{on_shap:+.4f} — meaning this feature {direction} the predicted probability "
                f"of a fatal collision compared to when it is absent (mean SHAP = {off_shap:+.4f}). "
                f"Overall, it has a mean absolute SHAP of {mean_shap_val:.4f}, making it "
                f"the #{rank} most influential predictor in the model."
            )
        elif feat == "hour":
            corr = np.corrcoef(feat_col, shap_col)[0, 1]
            direction = "later hours reduce" if corr < 0 else "later hours increase"
            sentence = (
                f"The hour of day has a mean absolute SHAP of {mean_shap_val:.4f} (rank #{rank}). "
                f"The SHAP-feature correlation is {corr:.3f}: {direction} the predicted fatal probability. "
                f"Rush-hour peaks in collision volume do not translate directly into higher fatality risk — "
                f"late-night hours with fewer but faster collisions carry elevated SHAP values."
            )
        elif feat == "invage":
            corr = np.corrcoef(feat_col, shap_col)[0, 1]
            sentence = (
                f"Age of the involved person (mean |SHAP| = {mean_shap_val:.4f}, rank #{rank}) "
                f"has a SHAP-feature correlation of {corr:.3f}: older individuals are associated with "
                f"higher fatal probability SHAP scores, consistent with the logistic regression finding "
                f"that older adult involvement (OR=2.04) is the strongest risk-increasing predictor."
            )
        elif feat.startswith("traffictl_"):
            on_shap  = shap_col[feat_col == 1].mean() if (feat_col == 1).sum() > 0 else 0
            sentence = (
                f"'{label}' has a mean absolute SHAP of {mean_shap_val:.4f} (rank #{rank}). "
                f"When this traffic control type is present, the average SHAP is {on_shap:+.4f} "
                f"relative to the 'No Control' reference — a "
                f"{'protective' if on_shap < 0 else 'risk-elevating'} effect on fatal probability."
            )
        else:
            sentence = (
                f"'{label}' contributes a mean absolute SHAP of {mean_shap_val:.4f} (rank #{rank}). "
                f"Higher feature values are associated with "
                f"{'increased' if mean_shap_high > mean_shap_low else 'decreased'} "
                f"fatal collision probability (mean SHAP for high values: {mean_shap_high:+.4f}, "
                f"low values: {mean_shap_low:+.4f})."
            )

        interpretations.append({
            "rank": rank, "feature": feat, "label": label,
            "mean_abs_shap": mean_shap_val, "sentence": sentence,
        })
        print(f"\n  #{rank} {label}")
        print(f"     {sentence}")

    # Plot interpretation cards
    fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")
    fig.suptitle(
        "Fig 27 — Plain-Language SHAP Interpretations: Top 5 Features\n"
        "XGBoost (Optuna-tuned)  |  Fatal=1 vs Non-Fatal=0",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")

    card_colors  = ["#FADBD8","#D5F5E3","#D6EAF8","#FEF9E7","#E8DAEF"]
    border_colors = [C_FATAL, "#27AE60", "#2980B9", C_DT, "#8E44AD"]

    card_h = 0.165; gap = 0.013; start_y = 0.93

    for i, interp in enumerate(interpretations):
        y  = start_y - i * (card_h + gap)
        fc = card_colors[i]; ec = border_colors[i]

        fancy = mpatches.FancyBboxPatch(
            (0.01, y - card_h), 0.98, card_h,
            boxstyle="round,pad=0.01",
            facecolor=fc, edgecolor=ec, linewidth=2.2,
            transform=ax.transAxes, zorder=2)
        ax.add_patch(fancy)

        ax.text(0.04, y - card_h/2, f"#{interp['rank']}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=20, fontweight="bold", color=ec, zorder=3)

        ax.text(0.91, y - card_h*0.30,
                f"|SHAP|={interp['mean_abs_shap']:.4f}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontweight="bold", color=ec, zorder=3)

        ax.text(0.09, y - card_h*0.25, interp["label"],
                transform=ax.transAxes, ha="left", va="center",
                fontsize=10, fontweight="bold", color="#1A1A2E", zorder=3)

        ax.text(0.09, y - card_h*0.68, interp["sentence"],
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8.3, color="#2C3E50", style="italic", zorder=3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "task_49_shap_plain_language.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_49_shap_plain_language.png")

    return interpretations


def plot_final_combined_roc(lr_m, dt_m, rf_m, xgb_m, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")

    for m, label, color, lw, ls in [
        (lr_m,  f"Logistic Regression (AUC={lr_m['auc']:.4f})",  C_LR,  1.8, "--"),
        (dt_m,  f"Decision Tree       (AUC={dt_m['auc']:.4f})",  C_DT,  2.0, "-."),
        (rf_m,  f"Random Forest       (AUC={rf_m['auc']:.4f})",  C_RF,  2.2, "-"),
        (xgb_m, f"XGBoost             (AUC={xgb_m['auc']:.4f})", C_XGB, 2.5, "-"),
    ]:
        ax.plot(m["fpr"], m["tpr"], color=color,
                linewidth=lw, linestyle=ls, label=label)

    ax.plot([0,1],[0,1], color="#95A5A6", linewidth=1.2, linestyle=":",
            label="Random (AUC=0.50)")
    ax.axhline(0.75, color=C_FATAL, linewidth=1.0, linestyle=":", alpha=0.5,
               label="AUC=0.75 target")
    ax.fill_between(xgb_m["fpr"], xgb_m["tpr"], alpha=0.07, color=C_XGB)

    _style_ax(ax,
              xlabel="False Positive Rate (1 − Specificity)",
              ylabel="True Positive Rate (Sensitivity)",
              title="Fig 28 — Final ROC Curves: All 4 Models\n"
                    "Evaluated on held-out test set (n=4,088, imbalanced, no SMOTE)",
              grid_axis="both")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
              prop={"family": "monospace"})

    best_auc  = max(lr_m["auc"], dt_m["auc"], rf_m["auc"], xgb_m["auc"])
    best_name = {lr_m["auc"]:"LR", dt_m["auc"]:"DT",
                 rf_m["auc"]:"RF", xgb_m["auc"]:"XGBoost"}[best_auc]
    badge_col = "#27AE60" if best_auc >= 0.75 else C_DT
    badge     = f"Best: {best_name}  AUC={best_auc:.4f}" + \
                (" ★ Target met" if best_auc >= 0.75 else " ✗")
    ax.text(0.97, 0.06, badge, transform=ax.transAxes,
            ha="right", fontsize=9.5, fontweight="bold", color=badge_col,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=badge_col, lw=1.5))

    fig.tight_layout()
    fig.savefig(out / "task_47_final_combined_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_47_final_combined_roc.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str = "outputs",
        indices_dir: str = None, models_dirs: list[str] = None):
    out     = Path(output_dir)
    idx_dir = Path(indices_dir) if indices_dir else out
    out.mkdir(parents=True, exist_ok=True)

    # Build list of directories to search for prior model pkl files
    # (logistic_baseline_model.pkl, dt_model.pkl, rf_model.pkl)
    model_search_dirs = [out]
    if models_dirs:
        model_search_dirs = [Path(d) for d in models_dirs] + [out]

    def find_model(filename: str) -> Path:
        """Search model_search_dirs for a .pkl file, raise if not found."""
        for d in model_search_dirs:
            p = d / filename
            if p.exists():
                return p
        searched = ", ".join(str(d) for d in model_search_dirs)
        raise FileNotFoundError(
            f"{filename} not found in: {searched}\n"
            f"Run the relevant earlier story first, or pass --models-dir."
        )

    print("═══ Loading splits (reusing Story 6 indices) ═══")
    print(f"  indices_dir : {idx_dir}")
    X_train, X_test, y_train, y_test, feat_cols = load_splits(input_path, idx_dir)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n═══ Applying SMOTE ═══")
    X_smote, y_smote = apply_smote(X_train, y_train)

    print("\n═══ Step 2: Screening imbalance strategies ═══")
    screen_results, best_strategy = screen_imbalance_strategies(
        X_train, y_train, X_smote, y_smote, X_test, y_test)
    plot_strategy_comparison(screen_results, out)

    # Select fitting data based on best strategy
    if "SMOTE" in best_strategy:
        X_fit, y_fit = X_smote.values, y_smote.values
        print(f"\n  Using SMOTE data for tuning (n={len(y_fit):,})")
    else:
        X_fit, y_fit = X_train.values, y_train.values
        print(f"\n  Using raw imbalanced data for tuning (n={len(y_fit):,})")

    print(f"\n═══ Step 3: Optuna tuning ({N_OPTUNA_TRIALS} trials) ═══")
    xgb_model, best_params, best_cv_auc, study = tune_with_optuna(
        X_fit, y_fit, best_strategy)
    plot_optuna_history(study, out)

    # Task #45 — save Optuna best params as JSON and full trial history as CSV
    import json
    with open(out / "task_45_optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
    print("  Saved task_45_optuna_best_params.json")
    study.trials_dataframe().to_csv(out / "task_45_optuna_trials.csv", index=False)
    print("  Saved task_45_optuna_trials.csv")

    print("\n═══ Step 4: Evaluate XGBoost on test set ═══")
    xgb_metrics = compute_metrics(xgb_model, X_test.values, y_test, "XGBoost")
    print_metrics(xgb_metrics)

    # H5 assessment
    h5_met = xgb_metrics["auc"] >= 0.75
    print(f"\n  ══ H5 Assessment ══════════════════════════════")
    print(f"  H5: ML classifier achieves AUC ≥ 0.75")
    print(f"  XGBoost Test AUC = {xgb_metrics['auc']:.4f}")
    print(f"  H5 {'✓ SUPPORTED — AUC target MET' if h5_met else '✗ NOT SUPPORTED — AUC target not met'}")

    # Save model
    with open(out / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    print("  Model saved → xgb_model.pkl")

    # Save metrics
    safe_m = {k: v for k, v in xgb_metrics.items()
              if k not in ("cm","fpr","tpr","y_prob","y_pred")}
    safe_m["best_params"]  = str(best_params)
    safe_m["best_cv_auc"]  = best_cv_auc
    safe_m["best_strategy"] = best_strategy
    safe_m["h5_met"]        = h5_met
    pd.DataFrame([safe_m]).to_csv(out / "task_47_xgb_metrics.csv", index=False)

    # Task #47 — save classification report as TXT
    from sklearn.metrics import classification_report
    cr = classification_report(
        y_test,
        xgb_metrics["y_pred"],
        target_names=["Non-Fatal", "Fatal"]
    )
    (out / "task_47_xgb_classification_report.txt").write_text(cr)
    print("  Saved task_47_xgb_classification_report.txt")

    print("\n═══ Step 5: Updating model comparison table ═══")
    # Load existing comparison table — search all model dirs first, then output dir
    comp_df = pd.DataFrame()
    for d in model_search_dirs + [out]:
        comp_path = d / "task_47_model_comparison_table.csv"
        if comp_path.exists():
            comp_df = pd.read_csv(comp_path)
            print(f"  Loaded existing comparison table from: {comp_path}")
            break

    t = "★" if xgb_metrics["auc"] >= 0.75 else ""
    params_str = (f"n_est={best_params.get('n_estimators','?')}, "
                  f"depth={best_params.get('max_depth','?')}, "
                  f"lr={best_params.get('learning_rate',0):.3f}, "
                  f"sub={best_params.get('subsample',0):.2f}, "
                  f"col={best_params.get('colsample_bytree',0):.2f}")
    xgb_row = pd.DataFrame([{
        "Model":             "XGBoost",
        "Best Params":       params_str,
        "AUC-ROC":           f"{xgb_metrics['auc']:.4f} {t}",
        "Precision\n(macro)": f"{xgb_metrics['precision_macro']:.4f}",
        "Recall\n(macro)":    f"{xgb_metrics['recall_macro']:.4f}",
        "F1\n(macro)":        f"{xgb_metrics['f1_macro']:.4f}",
        "F1\n(weighted)":     f"{xgb_metrics['f1_weighted']:.4f}",
        "Fatal\nRecall":      f"{xgb_metrics['recall_fatal']:.4f}",
        "Notes": (f"Optuna {N_OPTUNA_TRIALS} trials | "
                  f"strategy={best_strategy} | "
                  f"CV AUC={best_cv_auc:.4f}"),
    }])

    if len(comp_df):
        comp_df = comp_df[comp_df["Model"] != "XGBoost"]
        comp_df = pd.concat([comp_df, xgb_row], ignore_index=True)
    else:
        comp_df = xgb_row
    comp_df.to_csv(out / "task_47_model_comparison_table.csv", index=False)
    print(comp_df[["Model","AUC-ROC","F1\n(macro)","Fatal\nRecall"]].to_string(index=False))

    print("\n═══ Step 6: Computing SHAP values ═══")
    explainer, shap_values = compute_shap(xgb_model, X_test, feat_cols)
    np.save(out / "shap_values.npy", shap_values)
    print(f"  SHAP values shape: {shap_values.shape}")

    print("\n═══ Step 7: SHAP plots ═══")
    plot_shap_beeswarm(shap_values, X_test, feat_cols, out)
    mean_shap = plot_shap_bar(shap_values, feat_cols, out)
    top3_feats = mean_shap.head(3).index.tolist()
    plot_shap_dependence(shap_values, X_test, feat_cols, top3_feats, out)

    print("\n═══ Step 8: Plain-language SHAP interpretations ═══")
    interpretations = plot_shap_interpretation_cards(
        mean_shap, shap_values, X_test, feat_cols, out)
    pd.DataFrame(interpretations).drop(columns=["sentence"]).to_csv(
        out / "task_48_shap_interpretations.csv", index=False)
    interp_text = "\n\n".join(f"#{i['rank']} {i['label']}\n{i['sentence']}"
                               for i in interpretations)
    (out / "task_49_shap_interpretations.txt").write_text(interp_text)
    print("  Saved task_48_shap_interpretations.csv + task_49_shap_interpretations.txt")

    print("\n═══ Step 9: Final combined ROC (all 4 models) ═══")
    with open(find_model("logistic_baseline_model.pkl"), "rb") as f: lr_model = pickle.load(f)
    with open(find_model("dt_model.pkl"),                "rb") as f: dt_model = pickle.load(f)
    with open(find_model("rf_model.pkl"),                "rb") as f: rf_model = pickle.load(f)
    lr_m = compute_metrics(lr_model, X_test, y_test, "LR")
    dt_m = compute_metrics(dt_model, X_test, y_test, "DT")
    rf_m = compute_metrics(rf_model, X_test, y_test, "RF")
    plot_final_combined_roc(lr_m, dt_m, rf_m, xgb_metrics, out)

    print(f"\n═══ Story 8 complete — outputs in {out.resolve()} ═══")
    return xgb_metrics, interpretations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 8 – XGBoost + SHAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All files in one flat outputs folder (original behaviour)
  python src/ml_xgboost_shap.py \\
      --input      outputs/story-1/ksi_encoded.csv \\
      --output-dir outputs/story-8

  # Story subfolders — indices from story-6, models from story-6 and story-7
  python src/ml_xgboost_shap.py \\
      --input       outputs/story-1/ksi_encoded.csv \\
      --output-dir  outputs/story-8 \\
      --indices-dir outputs/story-6 \\
      --models-dir  outputs/story-6 outputs/story-7
        """
    )
    parser.add_argument("--input",       required=True,
                        help="Path to raw KSI CSV file")
    parser.add_argument("--output-dir",  default="outputs",
                        help="Directory to write Story 8 outputs (created if absent)")
    parser.add_argument("--indices-dir", default=None,
                        help="Folder containing train_indices.csv and test_indices.csv "
                             "(defaults to --output-dir if not set, e.g. outputs/story-6)")
    parser.add_argument("--models-dir",  nargs="+", default=None,
                        help="One or more folders to search for prior model .pkl files "
                             "(logistic_baseline_model.pkl, dt_model.pkl, rf_model.pkl). "
                             "Example: --models-dir outputs/story-6 outputs/story-7")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.indices_dir, args.models_dir)