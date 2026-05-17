"""
Story 4 – Logistic Regression + VIF Analysis
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5

Steps
─────
1.  Build feature matrix — imbalanced (no SMOTE) for interpretable coefficients
2.  First-pass VIF on raw 91-feature matrix — document inflation
3.  Resolve multicollinearity:
      • Use a curated 28-feature core set (one dummy per OHE group, drop reference)
      • Drop lat/lon, redundant temporal, cyclist (collinear with impactype)
4.  Second-pass VIF on core matrix — confirm all VIF < 5
5.  Fit statsmodels Logit (newton solver) on imbalanced training split
6.  Extract coefficient table: Predictor | Coef | SE | z | p | OR | 95% CI
7.  Highlight significant predictors (p < 0.05)
8.  Interpret top 5 in plain language
9.  Produce figures and CSVs

Usage:
    python src/logistic_regression.py \
        --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
        --output outputs/
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_preparation import (
    load_dataset, impute_and_flag, engineer_temporal,
    encode_target, encode_categoricals,
)

# ── Palette ───────────────────────────────────────────────────────────────────
C_SIG      = "#C0392B"
C_INSIG    = "#BDC3C7"
C_OR_LOW   = "#2980B9"
C_ACCENT   = "#E67E22"
C_BG       = "#F8F9FA"
C_GRID     = "#DEE2E6"
C_GREEN    = "#27AE60"
FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1A1A2E"}
FONT_AX    = {"fontsize": 10, "color": "#2C3E50"}
ALPHA      = 0.05

# ── Core feature set (28 predictors) ─────────────────────────────────────────
# Reference categories: light=Daylight, rdsfcond=Dry, traffictl=No Control,
#                       road_class=Major Arterial, accloc=At Intersection,
#                       impactype=Angle
CORE_FEATURES = [
    # Demographics
    "invage", "older_adult", "school_child",
    # Road users
    "pedestrian", "motorcyclist",
    # Behaviour
    "aggressive", "distracted", "red_light",
    # Temporal
    "hour", "is_weekend", "year",
    # Environmental (ref = Daylight / Dry)
    "light_Dark", "light_Dark with Artificial Lighting", "light_Dusk",
    "rdsfcond_Wet", "rdsfcond_Ice", "rdsfcond_Loose Snow",
    # Infrastructure (ref = No Control / Major Arterial / At Intersection)
    "traffictl_Traffic Signal", "traffictl_Stop Sign",
    "road_class_Expressway", "road_class_Local", "road_class_Minor Arterial",
    "accloc_Non-Intersection", "accloc_Intersection-Related",
    # Impact type (ref = Angle)
    "impactype_Pedestrian Collision(internal code)",
    "impactype_Cyclist Collision", "impactype_Rear End",
    "impactype_Turning Movement",
]

# ── All raw features (for first-pass VIF) ────────────────────────────────────
BASE_DROP = {
    "_id", "collision_id", "stname1", "stname2", "stname3",
    "geometry", "acclass", "acclass_binary", "accdate",
    "day_of_week_name", "month_name", "season",
    "injury", "drivact", "drivcond", "road_user",
    "wardname", "neighbourhood", "division",
    "pedact", "pedcond", "pedtype", "cyclistype", "cycact", "cyccond",
    "manoeuvre", "safequip", "vehtype", "initdir",
    "visible", "failtorem", "fatal_no", "per_inv", "veh_no", "per_no",
    "drivcond_missing", "aggressive_missing", "distracted_missing",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_matrices(input_path: str):
    df = load_dataset(input_path)
    df = impute_and_flag(df)
    df = engineer_temporal(df)
    df = encode_target(df)
    df = encode_categoricals(df)

    # Raw feature set (91 cols) — for VIF comparison only
    raw_cols = [c for c in df.columns
                if c not in BASE_DROP
                and pd.api.types.is_numeric_dtype(df[c])]
    X_raw = df[raw_cols].fillna(0)

    # Core feature set (28 cols) — for model
    # Remove persistent high-VIF cols: year collinear with hour/trend; pedestrian collinear with impactype_Pedestrian
    FINAL_DROP_VIF = ["year", "pedestrian", "impactype_Pedestrian Collision(internal code)"]
    core_cols = [c for c in CORE_FEATURES if c in df.columns and c not in FINAL_DROP_VIF]
    X_core = df[core_cols].fillna(0)

    y = df["acclass_binary"]

    # NOTE: This split is for STATISTICAL INFERENCE only (statsmodels Logit).
    # It uses the same seed and ratio as Story 6 but is independent —
    # no split indices are saved here. The ML train/test split and SMOTE
    # that feed Stories 6-9 live exclusively in ml_logistic_baseline.py.
    X_train_raw,  X_test_raw,  y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42)
    X_train_core, X_test_core, _,       _      = train_test_split(
        X_core, y, test_size=0.2, stratify=y, random_state=42)

    return (X_train_raw, X_train_core, X_test_core,
            y_train, y_test, raw_cols, core_cols)


# ─────────────────────────────────────────────────────────────────────────────
# VIF
# ─────────────────────────────────────────────────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    X_vals = X.values.astype(float)
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X_vals, i)
        except Exception:
            vif = np.nan
        rows.append({"feature": col, "VIF": round(float(vif), 3)})
    return (pd.DataFrame(rows)
              .sort_values("VIF", ascending=False)
              .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression
# ─────────────────────────────────────────────────────────────────────────────

def fit_logit(X_train: pd.DataFrame, y_train: pd.Series):
    X_const = sm.add_constant(X_train, has_constant="add")
    model   = sm.Logit(y_train, X_const)
    result  = model.fit(method="newton", maxiter=200, disp=False)
    return result


def extract_coef_table(result) -> pd.DataFrame:
    params   = result.params
    bse      = result.bse
    tvalues  = result.tvalues
    pvalues  = result.pvalues
    conf_int = result.conf_int()

    rows = []
    for pred in params.index:
        if pred == "const":
            continue
        coef  = params[pred]
        se    = bse[pred]
        z     = tvalues[pred]
        p     = pvalues[pred]
        ci_lo = conf_int.loc[pred, 0]
        ci_hi = conf_int.loc[pred, 1]
        rows.append({
            "Predictor":    pred,
            "Coefficient":  round(coef, 4),
            "Std Error":    round(se, 4),
            "z-statistic":  round(z, 4),
            "p-value":      p,
            "OR":           round(np.exp(coef), 4),
            "OR_CI_low":    round(np.exp(ci_lo), 4),
            "OR_CI_high":   round(np.exp(ci_hi), 4),
            "Significant":  bool(p < ALPHA),
        })

    return (pd.DataFrame(rows)
              .sort_values("p-value")
              .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language interpretations
# ─────────────────────────────────────────────────────────────────────────────

PLAIN_LABELS = {
    "invage":        "Each additional year of age",
    "older_adult":   "Involvement of an older adult (65+)",
    "school_child":  "Involvement of a school-age child",
    "pedestrian":    "Pedestrian involvement",
    "motorcyclist":  "Motorcyclist involvement",
    "aggressive":    "Aggressive driving flag",
    "distracted":    "Distracted driving flag",
    "red_light":     "Red-light violation",
    "hour":          "Each additional hour of day (0–23)",
    "is_weekend":    "Weekend collision (vs. weekday)",
    "year":          "Each additional year (trend)",
    "light_Dark":                              "Dark conditions (vs. Daylight)",
    "light_Dark with Artificial Lighting":     "Dark with artificial light (vs. Daylight)",
    "light_Dusk":                              "Dusk conditions (vs. Daylight)",
    "rdsfcond_Wet":                            "Wet road surface (vs. Dry)",
    "rdsfcond_Ice":                            "Icy road surface (vs. Dry)",
    "rdsfcond_Loose Snow":                     "Loose snow on road (vs. Dry)",
    "traffictl_Traffic Signal":                "Traffic signal control (vs. No Control)",
    "traffictl_Stop Sign":                     "Stop sign control (vs. No Control)",
    "road_class_Expressway":                   "Expressway (vs. Major Arterial)",
    "road_class_Local":                        "Local road (vs. Major Arterial)",
    "road_class_Minor Arterial":               "Minor arterial (vs. Major Arterial)",
    "accloc_Non-Intersection":                 "Non-intersection location (vs. At Intersection)",
    "accloc_Intersection-Related":             "Intersection-related location (vs. At Intersection)",
    "impactype_Pedestrian Collision(internal code)": "Pedestrian collision type (vs. Angle)",
    "impactype_Cyclist Collision":             "Cyclist collision type (vs. Angle)",
    "impactype_Rear End":                      "Rear-end collision (vs. Angle)",
    "impactype_Turning Movement":              "Turning-movement collision (vs. Angle)",
}


def plain_language_interpretations(tbl: pd.DataFrame) -> list[dict]:
    sig = tbl[tbl["Significant"]].copy()
    sig = sig[(sig["OR"] > 0.05) & (sig["OR"] < 100)]
    sig["or_distance"] = abs(np.log(sig["OR"]))
    top5 = sig.nlargest(5, "or_distance").reset_index(drop=True)

    interpretations = []
    for rank, row in top5.iterrows():
        pred    = row["Predictor"]
        OR      = row["OR"]
        ci_lo   = row["OR_CI_low"]
        ci_hi   = row["OR_CI_high"]
        p       = row["p-value"]
        p_str   = "< 0.001" if p < 0.001 else f"= {p:.4f}"
        label   = PLAIN_LABELS.get(pred, pred.replace("_", " "))
        ci_str  = f"[{ci_lo:.2f}–{ci_hi:.2f}]"

        if OR > 1:
            sentence = (
                f"{label} was associated with {OR:.2f}× higher odds of a fatal "
                f"collision compared to the reference category "
                f"(OR = {OR:.2f}, 95% CI = {ci_str}, p {p_str})."
            )
        else:
            pct = round((1 - OR) * 100, 1)
            sentence = (
                f"{label} was associated with a {pct}% reduction in the odds of a "
                f"fatal collision compared to the reference category "
                f"(OR = {OR:.2f}, 95% CI = {ci_str}, p {p_str})."
            )

        interpretations.append({
            "rank":     rank + 1,
            "predictor": pred,
            "label":    label,
            "OR":       OR,
            "CI_low":   ci_lo,
            "CI_high":  ci_hi,
            "p_value":  p,
            "sentence": sentence,
        })

    return interpretations


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel="", ylabel="", title="", grid_axis="y"):
    ax.set_facecolor(C_BG)
    ax.set_title(title, **FONT_TITLE, pad=8)
    ax.set_xlabel(xlabel, **FONT_AX)
    ax.set_ylabel(ylabel, **FONT_AX)
    ax.tick_params(labelsize=8.5, colors="#4A4A4A")
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)
    ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_vif_before_after(vif_before: pd.DataFrame,
                          vif_after: pd.DataFrame,
                          out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    fig.suptitle(
        "Fig 11 — VIF: Before (91 raw features) vs. After (28 core features)\n"
        "Threshold = 5  |  Resolved by curated feature selection + reference-category dropping",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )

    for ax, vif_df, title in [
        (axes[0], vif_before, "Before — Top 30 VIF (91 raw features)"),
        (axes[1], vif_after,  "After  — All 28 core features"),
    ]:
        plot_df = vif_df[np.isfinite(vif_df["VIF"])].head(30).copy()
        colors  = [C_SIG if v > 5 else C_GREEN for v in plot_df["VIF"]]
        labels  = [f[:38] for f in plot_df["feature"]]

        ax.barh(range(len(labels)), plot_df["VIF"].values,
                color=colors, edgecolor="white", zorder=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(5, color=C_ACCENT, linestyle="--", linewidth=1.5,
                   label="VIF = 5 threshold")

        for i, (bar_h, v) in enumerate(
                zip(plot_df["VIF"].values, plot_df["VIF"].values)):
            ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=7)

        _style_ax(ax, xlabel="VIF", title=title, grid_axis="x")
        ax.legend(fontsize=8.5, framealpha=0.9)

    legend_patches = [
        mpatches.Patch(color=C_SIG,   label="VIF > 5 (problematic)"),
        mpatches.Patch(color=C_GREEN, label="VIF ≤ 5 (acceptable)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9)

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out / "task_26_vif_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_26_vif_before_after.png")


def plot_coef_table(tbl: pd.DataFrame, result, out: Path) -> None:
    display = tbl.copy()
    display["p-value-str"] = display["p-value"].apply(
        lambda p: "< 0.001" if p < 0.001 else f"{p:.4f}")
    display["OR 95% CI"] = display.apply(
        lambda r: f"[{r['OR_CI_low']:.3f} – {r['OR_CI_high']:.3f}]", axis=1)
    display["Sig"] = display["Significant"].map({True: "★", False: ""})

    show_cols = ["Predictor", "Coefficient", "Std Error",
                 "z-statistic", "p-value-str", "OR", "OR 95% CI", "Sig"]
    cell_data = []
    for _, row in display[show_cols].iterrows():
        cell_data.append([
            row["Predictor"][:40],
            f"{row['Coefficient']:.4f}",
            f"{row['Std Error']:.4f}",
            f"{row['z-statistic']:.3f}",
            row["p-value-str"],
            f"{row['OR']:.4f}",
            row["OR 95% CI"],
            row["Sig"],
        ])

    n_sig = display["Significant"].sum()
    fig, ax = plt.subplots(figsize=(18, 13), facecolor="white")
    fig.suptitle(
        f"Fig 12 — Logistic Regression Coefficient Table  ({n_sig}/{len(display)} significant at α=0.05)\n"
        f"Pseudo R² = {result.prsquared:.4f}  |  AIC = {result.aic:.1f}  |  "
        f"Log-Likelihood = {result.llf:.1f}  |  n = {int(result.nobs):,}\n"
        "★ significant at p < 0.05  |  Reference: light=Daylight, rdsfcond=Dry, "
        "traffictl=No Control, road_class=Major Arterial, accloc=At Intersection, impactype=Angle",
        fontsize=10, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")

    col_labels = ["Predictor", "Coeff", "Std Err", "z",
                  "p-value", "OR", "95% CI", "Sig"]
    tbl_obj = ax.table(cellText=cell_data, colLabels=col_labels,
                       cellLoc="center", loc="center")
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(8.5)
    tbl_obj.scale(1, 1.72)
    tbl_obj.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl_obj[0, j].set_facecolor("#1A1A2E")
        tbl_obj[0, j].set_text_props(color="white", fontweight="bold")

    for i, (_, row) in enumerate(display.iterrows(), start=1):
        row_color = "#FADBD8" if row["Significant"] else "#F2F3F4"
        for j in range(len(col_labels)):
            tbl_obj[i, j].set_facecolor(row_color)
        if row["Significant"]:
            tbl_obj[i, len(col_labels)-1].set_text_props(color=C_SIG, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#FADBD8", label="Significant (p < 0.05)"),
        mpatches.Patch(color="#F2F3F4", label="Not significant"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=9, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out / "task_25_logit_coef_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_25_logit_coef_table.png")


def plot_or_forest(tbl: pd.DataFrame, out: Path) -> None:
    sig = tbl[tbl["Significant"]].copy()
    sig = sig[(sig["OR"] > 0.05) & (sig["OR"] < 100)]
    sig = sig.sort_values("OR", ascending=True).reset_index(drop=True)

    labels = [PLAIN_LABELS.get(p, p.replace("_", " "))[:55]
              for p in sig["Predictor"]]
    ors   = sig["OR"].values
    ci_lo = sig["OR_CI_low"].values
    ci_hi = sig["OR_CI_high"].values
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(
        figsize=(12, max(6, len(labels) * 0.42)), facecolor="white")
    fig.suptitle(
        "Fig 13 — Forest Plot: Odds Ratios for All Significant Predictors (p < 0.05)\n"
        "(log scale  |  OR > 1 = increased fatal odds  |  CI bars = 95% confidence interval)",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )

    colors = [C_SIG if o > 1 else C_OR_LOW for o in ors]
    ax.barh(y_pos, ors, color=colors, alpha=0.7,
            edgecolor="white", height=0.55, zorder=3)
    ax.errorbar(ors, y_pos,
                xerr=[ors - ci_lo, ci_hi - ors],
                fmt="none", color="#1A1A2E",
                capsize=3, linewidth=1.2, capthick=1.2, zorder=4)
    ax.axvline(1.0, color=C_ACCENT, linestyle="--",
               linewidth=2.0, label="OR = 1.0 (null)", zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Odds Ratio (log scale)", **FONT_AX)
    ax.set_xscale("log")
    ax.set_facecolor(C_BG)
    ax.grid(axis="x", color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)

    legend_patches = [
        mpatches.Patch(color=C_SIG,    label="OR > 1 — increased fatal odds"),
        mpatches.Patch(color=C_OR_LOW, label="OR < 1 — reduced fatal odds"),
        plt.Line2D([0], [0], color=C_ACCENT, linestyle="--",
                   lw=2, label="OR = 1.0 (no effect)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, framealpha=0.9,
              loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "task_27_top_predictors_or_plot.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_27_top_predictors_or_plot.png")


def plot_plain_language(interpretations: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")
    fig.suptitle(
        "Fig 14 — Plain-Language Interpretation of Top 5 Significant Predictors\n"
        "Logistic Regression  |  Fatal Injury = 1, Non-Fatal = 0  |  "
        "Reference categories noted in each card",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")

    card_colors  = ["#FADBD8", "#D5F5E3", "#D6EAF8", "#FEF9E7", "#E8DAEF"]
    border_colors = [C_SIG, C_GREEN, "#2980B9", C_ACCENT, "#8E44AD"]

    card_h = 0.168
    gap    = 0.012
    start_y = 0.93

    for i, interp in enumerate(interpretations):
        y  = start_y - i * (card_h + gap)
        fc = card_colors[i]
        ec = border_colors[i]

        fancy = mpatches.FancyBboxPatch(
            (0.01, y - card_h), 0.98, card_h,
            boxstyle="round,pad=0.01",
            facecolor=fc, edgecolor=ec, linewidth=2.2,
            transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(fancy)

        # Rank
        ax.text(0.04, y - card_h / 2, f"#{interp['rank']}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=20, fontweight="bold", color=ec, zorder=3)

        # OR + CI + p
        or_val = interp["OR"]
        ax.text(0.91, y - card_h * 0.28,
                f"OR = {or_val:.2f}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ec, zorder=3)
        ax.text(0.91, y - card_h * 0.57,
                f"95% CI [{interp['CI_low']:.2f} – {interp['CI_high']:.2f}]",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#4A4A4A", zorder=3)
        p_str = "p < 0.001" if interp["p_value"] < 0.001 else f"p = {interp['p_value']:.4f}"
        ax.text(0.91, y - card_h * 0.82,
                p_str,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#4A4A4A", style="italic", zorder=3)

        # Label
        ax.text(0.09, y - card_h * 0.25,
                interp["label"],
                transform=ax.transAxes, ha="left", va="center",
                fontsize=10, fontweight="bold", color="#1A1A2E", zorder=3)

        # Sentence
        ax.text(0.09, y - card_h * 0.70,
                interp["sentence"],
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8.8, color="#2C3E50", style="italic", zorder=3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "task_27_plain_language_interpretations.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_27_plain_language_interpretations.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str = "outputs"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("═══ Step 1: Building feature matrices ═══")
    (X_train_raw, X_train_core, X_test_core,
     y_train, y_test, raw_cols, core_cols) = build_matrices(input_path)
    print(f"  Raw matrix:  {X_train_raw.shape[0]:,} × {X_train_raw.shape[1]}")
    print(f"  Core matrix: {X_train_core.shape[0]:,} × {X_train_core.shape[1]}")
    print(f"  Fatal in train: {y_train.sum():,} ({y_train.mean()*100:.1f}%) — NO SMOTE")

    print("\n═══ Step 2: First-pass VIF (91 raw features) ═══")
    vif_before = compute_vif(X_train_raw)
    n_inf   = (vif_before["VIF"] == np.inf).sum()
    n_high  = (vif_before["VIF"] > 5).sum()
    print(f"  Infinite VIF (perfect collinearity): {n_inf}")
    print(f"  VIF > 5: {n_high}")
    print("  Top 10:")
    finite_vif = vif_before[np.isfinite(vif_before["VIF"])]
    print(finite_vif.head(10).to_string(index=False))

    print("\n═══ Step 3: Resolution strategy ═══")
    print("  → Selected 28-feature core set:")
    print("    • Dropped all OHE reference categories (one per group)")
    print("    • Dropped latitude/longitude (spatial collinearity, not interpretable in logit)")
    print("    • Dropped 'cyclist' boolean (collinear with impactype_Cyclist Collision)")
    print("    • Dropped 'day_of_week' (collinear with is_weekend)")
    print("    • Dropped 'month' (cyclic, collinear with year/hour)")
    print("    • Dropped sparse OHE dummies (< 50 records in category)")
    print(f"  → Core features used: {core_cols}")

    print("\n═══ Step 4: Second-pass VIF (28 core features) ═══")
    vif_after = compute_vif(X_train_core)
    high_after = vif_after[vif_after["VIF"] > 5]
    if len(high_after):
        print(f"  WARNING — {len(high_after)} features still > 5:")
        print(high_after.to_string(index=False))
    else:
        print(f"  ✓ All {len(vif_after)} features have VIF ≤ 5")
    print(vif_after.to_string(index=False))

    # Save VIF reports
    vif_before["pass"] = "before (91 features)"
    vif_after["pass"]  = "after (28 core features)"
    pd.concat([vif_before, vif_after]).to_csv(out / "task_26_vif_report.csv", index=False)
    print("  Saved task_26_vif_report.csv")

    print("\n═══ Step 5: Fitting logistic regression (Newton-Raphson) ═══")
    result = fit_logit(X_train_core, y_train)
    print(f"  Converged:      {result.mle_retvals.get('converged', True)}")
    print(f"  n (train):      {int(result.nobs):,}")
    print(f"  Log-likelihood: {result.llf:.4f}")
    print(f"  AIC:            {result.aic:.4f}")
    print(f"  McFadden R²:    {result.prsquared:.4f}")

    # Task #24 — save model summary
    with open(out / "task_24_logistic_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(result.summary()))
        f.write(f"\n\nModel fit statistics:\n")
        f.write(f"  Converged      : {result.mle_retvals.get('converged', True)}\n")
        f.write(f"  n (train)      : {int(result.nobs):,}\n")
        f.write(f"  Log-likelihood : {result.llf:.4f}\n")
        f.write(f"  AIC            : {result.aic:.4f}\n")
        f.write(f"  McFadden R²    : {result.prsquared:.4f}\n")
    print("  Saved task_24_logistic_model_summary.txt")

    print("\n═══ Step 6: Coefficient table ═══")
    coef_tbl = extract_coef_table(result)
    n_sig = coef_tbl["Significant"].sum()
    print(f"  {n_sig}/{len(coef_tbl)} predictors significant at α = 0.05")
    print(coef_tbl[["Predictor","Coefficient","OR","p-value",
                     "OR_CI_low","OR_CI_high","Significant"]].to_string(index=False))
    coef_tbl.to_csv(out / "task_25_logit_results.csv", index=False)
    print("  Saved task_25_logit_results.csv")

    print("\n═══ Step 8: Plain-language interpretations (top 5) ═══")
    interps = plain_language_interpretations(coef_tbl)
    for interp in interps:
        print(f"\n  #{interp['rank']}  {interp['predictor']}")
        print(f"     {interp['sentence']}")

    print("\n═══ Generating figures ═══")
    plot_vif_before_after(vif_before, vif_after, out)
    plot_coef_table(coef_tbl, result, out)
    plot_or_forest(coef_tbl, out)
    plot_plain_language(interps, out)

    print(f"\n═══ Complete — outputs saved to {out.resolve()} ═══")
    return coef_tbl, interps, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 4 – Logistic Regression + VIF")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.input, args.output_dir)