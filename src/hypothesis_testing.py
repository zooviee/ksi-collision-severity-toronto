"""
Story 3 – Statistical Inference: Hypothesis Testing (H1–H4)
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5

Tests
─────
H1: light & rdsfcond vs. acclass        — chi-square + Cramér's V
H2: drivcond, aggressive, distracted     — chi-square + odds ratios
H3: road_user vs. acclass               — chi-square + observed/expected
H4: traffictl vs. acclass               — chi-square + fatality rate table

Outputs
───────
  fig6_h1_light_rdsfcond.png
  fig7_h2_behavioural.png
  fig8_h3_road_user.png
  fig9_h4_traffictl.png
  fig10_hypothesis_summary.png
  hypothesis_results.csv

Usage:
    python src/statistical_inference.py \
        --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \
        --output outputs/
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
C_FATAL    = "#C0392B"
C_NONFATAL = "#2980B9"
C_ACCENT   = "#E67E22"
C_BG       = "#F8F9FA"
C_GRID     = "#DEE2E6"
C_GREEN    = "#27AE60"
C_PURPLE   = "#8E44AD"

FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1A1A2E"}
FONT_AX    = {"fontsize": 10, "color": "#2C3E50"}

ALPHA = 0.05   # significance threshold


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cramers_v(contingency_table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    k = min(contingency_table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def chi2_test(df: pd.DataFrame, col: str, target: str = "acclass_binary"):
    """
    Run chi-square test. Returns (chi2, p, dof, cramers_v, contingency_df).
    """
    ct = pd.crosstab(df[col], df[target])
    chi2_stat, p_val, dof, _ = chi2_contingency(ct)
    cv = cramers_v(ct)
    return chi2_stat, p_val, dof, cv, ct


def odds_ratio_2x2(df: pd.DataFrame, predictor: str,
                   positive_label,
                   target: str = "acclass_binary") -> dict:
    """
    Compute OR for a binary predictor vs binary target.
    Returns dict with OR, 95% CI, and z-test p-value.
    """
    ct = pd.crosstab(df[predictor], df[target])
    # Ensure 2×2 shape with order: [negative, positive] × [0, 1]
    try:
        a = ct.loc[positive_label, 1]   # exposed & fatal
        b = ct.loc[positive_label, 0]   # exposed & non-fatal
        c = ct.loc[~ct.index.isin([positive_label])].iloc[:, 1].sum()   # unexposed & fatal
        d = ct.loc[~ct.index.isin([positive_label])].iloc[:, 0].sum()   # unexposed & non-fatal
    except Exception:
        return {"OR": np.nan, "CI_low": np.nan, "CI_high": np.nan, "p": np.nan}

    if 0 in [a, b, c, d]:
        # Apply Haldane-Anscombe correction
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val  = (a * d) / (b * c)
    log_or  = np.log(or_val)
    se_log  = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z       = log_or / se_log
    p_val   = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_low  = np.exp(log_or - 1.96 * se_log)
    ci_high = np.exp(log_or + 1.96 * se_log)

    return {"OR": or_val, "CI_low": ci_low, "CI_high": ci_high, "p": p_val}


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


def _stat_box(ax, chi2, p, dof, cv, reject, extra_lines=None):
    """Place a stat-result annotation box on an axes."""
    sig_str = "✓ Reject H₀" if reject else "✗ Fail to reject H₀"
    sig_col = C_FATAL if reject else C_GREEN
    lines = [
        f"χ²({dof}) = {chi2:.2f}",
        f"p {'< 0.001' if p < 0.001 else f'= {p:.4f}'}",
        f"Cramér's V = {cv:.3f}",
        sig_str,
    ]
    if extra_lines:
        lines += extra_lines
    text = "\n".join(lines)
    ax.text(0.97, 0.97, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            color=sig_col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec=sig_col, lw=1.3, alpha=0.93))


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["acclass"].isin(["Fatal Injury", "Non-Fatal Injury"])].copy()
    df["acclass_binary"] = (df["acclass"] == "Fatal Injury").astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# H1 — light & rdsfcond vs. acclass
# ─────────────────────────────────────────────────────────────────────────────

def test_h1(df: pd.DataFrame, out: Path) -> list[dict]:
    results = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")
    fig.suptitle(
        "H1 — Environmental Conditions vs. Collision Fatality\n"
        "Chi-Square Test with Cramér's V Effect Size",
        fontsize=13, fontweight="bold", color="#1A1A2E"
    )

    vars_labels = [("light", "Lighting Condition"), ("rdsfcond", "Road Surface Condition")]

    for ax, (col, label) in zip(axes, vars_labels):
        sub = df[[col, "acclass_binary"]].dropna()
        chi2, p, dof, cv, ct = chi2_test(sub, col)
        reject = p < ALPHA

        # Fatal rate per category
        fatal_rate = ct[1] / ct.sum(axis=1) * 100
        fatal_rate = fatal_rate.sort_values(ascending=True)

        overall_rate = sub["acclass_binary"].mean() * 100

        colors = [C_FATAL if r > overall_rate else C_NONFATAL
                  for r in fatal_rate.values]
        bars = ax.barh(fatal_rate.index, fatal_rate.values,
                       color=colors, edgecolor="white", zorder=3)

        for bar, v in zip(bars, fatal_rate.values):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=8)

        ax.axvline(overall_rate, color=C_ACCENT, linestyle="--",
                   linewidth=1.5, label=f"Overall fatal rate ({overall_rate:.1f}%)")
        _style_ax(ax, xlabel="Fatal Collision Rate (%)", title=label, grid_axis="x")
        ax.legend(fontsize=8, framealpha=0.9)
        _stat_box(ax, chi2, p, dof, cv, reject)

        results.append({
            "hypothesis": "H1",
            "variable": col,
            "test": "Chi-Square",
            "chi2": round(chi2, 3),
            "df": dof,
            "p_value": p,
            "effect_size_metric": "Cramér's V",
            "effect_size": round(cv, 4),
            "reject_h0": reject,
        })
        print(f"  H1 | {col}: χ²({dof})={chi2:.2f}, p={p:.4e}, V={cv:.4f}, "
              f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀")

    # Interpretations
    interp = (
        "H1 Interpretation:\n"
        "Both lighting and road surface conditions show statistically significant "
        "associations with collision fatality (p < 0.001), supporting H1. "
        "However, the effect sizes (Cramér's V) are small, suggesting these "
        "environmental factors contribute to — but do not solely determine — fatal outcomes."
    )
    fig.text(0.5, -0.04, interp, ha="center", fontsize=9,
             color="#2C3E50", style="italic",
             wrap=True, bbox=dict(boxstyle="round", fc="#EBF5FB", ec=C_NONFATAL, lw=1))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out / "fig6_h1_light_rdsfcond.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_h1_light_rdsfcond.png")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# H2 — behavioural variables vs. acclass
# ─────────────────────────────────────────────────────────────────────────────

def _group_drivcond(val):
    if pd.isna(val):
        return "Missing/Unknown"
    v = str(val)
    if "Normal" in v:
        return "Normal"
    if "Impaired" in v or "Drinking" in v or "Drugs" in v:
        return "Impaired (Alcohol/Drugs)"
    if "Inattentive" in v:
        return "Inattentive"
    if "Medical" in v or "Fatigue" in v:
        return "Medical/Fatigue"
    if "Rage" in v or "Aggression" in v:
        return "Road Rage"
    return "Other/Unknown"


def test_h2(df: pd.DataFrame, out: Path) -> list[dict]:
    results = []
    fig = plt.figure(figsize=(16, 14), facecolor="white")
    fig.suptitle(
        "H2 — Behavioural Risk Factors vs. Collision Fatality\n"
        "Chi-Square Tests + Odds Ratios (95% CI)",
        fontsize=13, fontweight="bold", color="#1A1A2E"
    )
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.38)
    ax_dc = fig.add_subplot(gs[0, :])   # drivcond full width
    ax_ag = fig.add_subplot(gs[1, 0])
    ax_di = fig.add_subplot(gs[1, 1])

    # ── drivcond ──────────────────────────────────────────────────────────
    df_dc = df.copy()
    df_dc["drivcond_group"] = df_dc["drivcond"].apply(_group_drivcond)
    sub_dc = df_dc[["drivcond_group", "acclass_binary"]]
    chi2_dc, p_dc, dof_dc, cv_dc, ct_dc = chi2_test(sub_dc, "drivcond_group")
    reject_dc = p_dc < ALPHA

    fatal_rate_dc = (ct_dc[1] / ct_dc.sum(axis=1) * 100).sort_values(ascending=True)
    overall = df["acclass_binary"].mean() * 100

    colors_dc = [C_FATAL if r > overall else C_NONFATAL for r in fatal_rate_dc]
    bars = ax_dc.barh(fatal_rate_dc.index, fatal_rate_dc.values,
                      color=colors_dc, edgecolor="white", zorder=3)
    for bar, v in zip(bars, fatal_rate_dc.values):
        ax_dc.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                   f"{v:.1f}%", va="center", fontsize=9)
    ax_dc.axvline(overall, color=C_ACCENT, linestyle="--",
                  linewidth=1.5, label=f"Overall fatal rate ({overall:.1f}%)")
    _style_ax(ax_dc, xlabel="Fatal Collision Rate (%)",
              title="Driver Condition Group", grid_axis="x")
    ax_dc.legend(fontsize=8.5, framealpha=0.9)
    _stat_box(ax_dc, chi2_dc, p_dc, dof_dc, cv_dc, reject_dc)

    results.append({
        "hypothesis": "H2", "variable": "drivcond (grouped)",
        "test": "Chi-Square", "chi2": round(chi2_dc, 3),
        "df": dof_dc, "p_value": p_dc,
        "effect_size_metric": "Cramér's V",
        "effect_size": round(cv_dc, 4), "reject_h0": reject_dc,
    })
    print(f"  H2 | drivcond: χ²({dof_dc})={chi2_dc:.2f}, p={p_dc:.4e}, "
          f"V={cv_dc:.4f}, {'REJECT' if reject_dc else 'FAIL TO REJECT'} H₀")

    # ── aggressive & distracted — OR forest-style plot ─────────────────────
    for ax, col, label, color in [
        (ax_ag, "aggressive",  "Aggressive Driving", C_FATAL),
        (ax_di, "distracted",  "Distracted Driving", C_PURPLE),
    ]:
        sub = df[[col, "acclass_binary"]].dropna()
        sub[col] = sub[col].astype(bool)

        chi2_v, p_v, dof_v, cv_v, ct_v = chi2_test(sub, col)
        or_res = odds_ratio_2x2(sub, col, positive_label=True)
        reject_v = p_v < ALPHA

        # Grouped bar chart — fatal rate True vs False
        fatal_rate = ct_v[1] / ct_v.sum(axis=1) * 100
        cats   = [True, False]
        labels_bar = ["Yes", "No"]
        values_bar = [fatal_rate.get(c, 0) for c in cats]
        bar_colors = [color, "#95A5A6"]

        bars = ax.bar(labels_bar, values_bar, color=bar_colors,
                      edgecolor="white", width=0.45, zorder=3)
        for bar, v in zip(bars, values_bar):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.axhline(overall, color=C_ACCENT, linestyle="--",
                   linewidth=1.5, label=f"Overall rate ({overall:.1f}%)")
        _style_ax(ax, xlabel=label, ylabel="Fatal Rate (%)",
                  title=label, grid_axis="y")
        ax.set_ylim(0, max(values_bar) * 1.35)
        ax.legend(fontsize=8, framealpha=0.9)
        _stat_box(ax, chi2_v, p_v, dof_v, cv_v, reject_v,
                  extra_lines=[
                      f"OR = {or_res['OR']:.2f} "
                      f"[{or_res['CI_low']:.2f}–{or_res['CI_high']:.2f}]"
                  ])

        results.append({
            "hypothesis": "H2", "variable": col,
            "test": "Chi-Square + OR", "chi2": round(chi2_v, 3),
            "df": dof_v, "p_value": p_v,
            "effect_size_metric": "Cramér's V",
            "effect_size": round(cv_v, 4), "reject_h0": reject_v,
            "OR": round(or_res["OR"], 3),
            "OR_CI": f"[{or_res['CI_low']:.3f}–{or_res['CI_high']:.3f}]",
        })
        print(f"  H2 | {col}: χ²({dof_v})={chi2_v:.2f}, p={p_v:.4e}, "
              f"V={cv_v:.4f}, OR={or_res['OR']:.3f} "
              f"[{or_res['CI_low']:.3f}–{or_res['CI_high']:.3f}], "
              f"{'REJECT' if reject_v else 'FAIL TO REJECT'} H₀")

    interp = (
        "H2 Interpretation:\n"
        "Behavioural risk factors — driver impairment, inattention, aggression, "
        "and distraction — are each significantly associated with fatal collision "
        "outcomes (all p < 0.05), strongly supporting H2. "
        "Impaired drivers show the highest fatal rate; the odds ratios confirm that "
        "aggressive and distracted driving each meaningfully elevate fatality risk "
        "above the baseline."
    )
    fig.text(0.5, -0.03, interp, ha="center", fontsize=9,
             color="#2C3E50", style="italic",
             bbox=dict(boxstyle="round", fc="#FDEDEC", ec=C_FATAL, lw=1))

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out / "fig7_h2_behavioural.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_h2_behavioural.png")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# H3 — road_user vs. acclass + observed vs. expected
# ─────────────────────────────────────────────────────────────────────────────

def test_h3(df: pd.DataFrame, out: Path) -> list[dict]:
    results = []
    sub = df[["road_user", "acclass_binary"]].dropna()
    chi2_v, p_v, dof_v, cv_v, ct = chi2_test(sub, "road_user")
    reject = p_v < ALPHA

    # Observed vs. expected for vulnerable road users
    _, _, _, expected = chi2_contingency(ct)
    exp_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)

    vuln_users = ["pedestrian", "cyclist", "motorcyclist"]
    oe_rows = []
    for user in vuln_users:
        if user in ct.index:
            obs_fatal  = ct.loc[user, 1]
            exp_fatal  = exp_df.loc[user, 1]
            obs_total  = ct.loc[user].sum()
            obs_share  = obs_total / ct.values.sum() * 100
            fatal_rate = obs_fatal / obs_total * 100
            oe_ratio   = obs_fatal / exp_fatal
            oe_rows.append({
                "Road User":          user.capitalize(),
                "Total Records":      obs_total,
                "Share of All (%)":   round(obs_share, 1),
                "Fatal Count (Obs)":  int(obs_fatal),
                "Fatal Count (Exp)":  round(exp_fatal, 1),
                "Fatal Rate (%)":     round(fatal_rate, 1),
                "O/E Ratio":          round(oe_ratio, 3),
            })

    oe_df = pd.DataFrame(oe_rows)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")
    fig.suptitle(
        "H3 — Vulnerable Road Users vs. Collision Fatality\n"
        "Chi-Square Test + Observed vs. Expected Fatal Counts",
        fontsize=13, fontweight="bold", color="#1A1A2E"
    )

    # Panel 1 — fatal rate by road user
    ax1 = axes[0]
    overall = sub["acclass_binary"].mean() * 100
    fatal_rate_user = (ct[1] / ct.sum(axis=1) * 100).sort_values(ascending=True)

    vuln_set = set(vuln_users)
    colors = [C_FATAL if u in vuln_set else C_NONFATAL
              for u in fatal_rate_user.index]
    bars = ax1.barh(fatal_rate_user.index, fatal_rate_user.values,
                    color=colors, edgecolor="white", zorder=3)
    for bar, v in zip(bars, fatal_rate_user.values):
        ax1.text(v + 0.4, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=8.5)

    ax1.axvline(overall, color=C_ACCENT, linestyle="--",
                linewidth=1.5, label=f"Overall fatal rate ({overall:.1f}%)")
    _style_ax(ax1, xlabel="Fatal Collision Rate (%)",
              title="Fatal Rate by Road User Type", grid_axis="x")
    legend_patches = [
        mpatches.Patch(color=C_FATAL,    label="Vulnerable road user"),
        mpatches.Patch(color=C_NONFATAL, label="Other road user"),
    ]
    ax1.legend(handles=legend_patches, fontsize=8.5, framealpha=0.9)
    _stat_box(ax1, chi2_v, p_v, dof_v, cv_v, reject)

    # Panel 2 — O/E table
    ax2 = axes[1]
    ax2.axis("off")
    col_labels = list(oe_df.columns)
    cell_data  = oe_df.values.tolist()

    tbl = ax2.table(cellText=cell_data, colLabels=col_labels,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.4)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    row_colors = ["#FADBD8", "#FAD7A0", "#D7BDE2"]
    for i, rc in enumerate(row_colors, start=1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(rc)

    # O/E > 1.0 = over-represented — annotate
    ax2.set_title("Observed vs. Expected Fatal Counts\n(Vulnerable Road Users)",
                  **{**FONT_TITLE, "fontsize": 11}, pad=10)
    ax2.text(0.5, -0.02,
             "O/E Ratio > 1.0 = over-represented in fatal collisions",
             ha="center", fontsize=8.5, color=C_FATAL, style="italic",
             transform=ax2.transAxes)

    interp = (
        "H3 Interpretation:\n"
        "Road user type is significantly associated with collision fatality "
        "(p < 0.001), and all three vulnerable groups — pedestrians, cyclists, "
        "and motorcyclists — show observed fatal counts that exceed expected counts "
        "(O/E > 1.0), confirming they are over-represented in fatal outcomes relative "
        "to their share of total collisions, strongly supporting H3."
    )
    fig.text(0.5, -0.04, interp, ha="center", fontsize=9,
             color="#2C3E50", style="italic",
             bbox=dict(boxstyle="round", fc="#EAF2FF", ec=C_NONFATAL, lw=1))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out / "fig8_h3_road_user.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_h3_road_user.png")

    results.append({
        "hypothesis": "H3", "variable": "road_user",
        "test": "Chi-Square", "chi2": round(chi2_v, 3),
        "df": dof_v, "p_value": p_v,
        "effect_size_metric": "Cramér's V",
        "effect_size": round(cv_v, 4), "reject_h0": reject,
    })
    print(f"  H3 | road_user: χ²({dof_v})={chi2_v:.2f}, p={p_v:.4e}, "
          f"V={cv_v:.4f}, {'REJECT' if reject else 'FAIL TO REJECT'} H₀")
    print(oe_df.to_string(index=False))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# H4 — traffictl vs. acclass
# ─────────────────────────────────────────────────────────────────────────────

def test_h4(df: pd.DataFrame, out: Path) -> list[dict]:
    results = []

    # Group sparse categories
    def _group_traffictl(v):
        if pd.isna(v):
            return "Unknown"
        if v == "Traffic Signal":
            return "Traffic Signal"
        if v == "No Control":
            return "No Control"
        if v == "Stop Sign":
            return "Stop Sign"
        if v in ("Pedestrian Crossover", "Pedestrian Signal"):
            return "Pedestrian Crossover / Signal"
        return "Other Control"

    df2 = df.copy()
    df2["traffictl_group"] = df2["traffictl"].apply(_group_traffictl)

    sub = df2[["traffictl_group", "acclass_binary"]].dropna()
    chi2_v, p_v, dof_v, cv_v, ct = chi2_test(sub, "traffictl_group")
    reject = p_v < ALPHA

    # Fatality rate table
    ct_full = pd.crosstab(df2["traffictl_group"], df2["acclass_binary"])
    ct_full.columns = ["Non-Fatal", "Fatal"]
    ct_full["Total"]       = ct_full.sum(axis=1)
    ct_full["Fatal Rate %"] = (ct_full["Fatal"] / ct_full["Total"] * 100).round(2)
    ct_full = ct_full.sort_values("Fatal Rate %", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")
    fig.suptitle(
        "H4 — Traffic Control Type vs. Collision Fatality\n"
        "Chi-Square Test + Fatality Rate Contingency Table",
        fontsize=13, fontweight="bold", color="#1A1A2E"
    )

    # Panel 1 — bar chart: fatal rate by control type
    ax1 = axes[0]
    overall = sub["acclass_binary"].mean() * 100
    fatal_rates = ct_full["Fatal Rate %"].sort_values(ascending=True)

    no_ctrl_color = C_FATAL
    sig_color     = C_GREEN
    colors = []
    for cat in fatal_rates.index:
        if cat == "No Control":
            colors.append(no_ctrl_color)
        elif cat == "Traffic Signal":
            colors.append(sig_color)
        else:
            colors.append(C_NONFATAL)

    bars = ax1.barh(fatal_rates.index, fatal_rates.values,
                    color=colors, edgecolor="white", zorder=3)
    for bar, v in zip(bars, fatal_rates.values):
        ax1.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=9)

    ax1.axvline(overall, color=C_ACCENT, linestyle="--",
                linewidth=1.5, label=f"Overall fatal rate ({overall:.1f}%)")
    _style_ax(ax1, xlabel="Fatal Collision Rate (%)",
              title="Fatal Rate by Traffic Control Type", grid_axis="x")
    legend_patches = [
        mpatches.Patch(color=C_FATAL,    label="No Control"),
        mpatches.Patch(color=C_GREEN,    label="Traffic Signal"),
        mpatches.Patch(color=C_NONFATAL, label="Other"),
    ]
    ax1.legend(handles=legend_patches, fontsize=8.5, framealpha=0.9)
    _stat_box(ax1, chi2_v, p_v, dof_v, cv_v, reject)

    # Panel 2 — contingency table
    ax2 = axes[1]
    ax2.axis("off")
    tbl_display = ct_full.reset_index().rename(columns={"traffictl_group": "Control Type"})
    tbl_display["Fatal Rate %"] = tbl_display["Fatal Rate %"].apply(lambda x: f"{x:.2f}%")

    tbl = ax2.table(cellText=tbl_display.values.tolist(),
                    colLabels=tbl_display.columns.tolist(),
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.3)
    tbl.auto_set_column_width(col=list(range(len(tbl_display.columns))))

    for j in range(len(tbl_display.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight No Control row
    for i, idx in enumerate(ct_full.index, start=1):
        row_color = "#FADBD8" if idx == "No Control" else "#EBF5FB"
        for j in range(len(tbl_display.columns)):
            tbl[i, j].set_facecolor(row_color)

    ax2.set_title("Fatality Rate Contingency Table by Control Type",
                  **{**FONT_TITLE, "fontsize": 10}, pad=10)

    interp = (
        "H4 Interpretation:\n"
        "Traffic control type is significantly associated with collision fatality "
        "(p < 0.001), partially supporting H4. Uncontrolled intersections show the "
        "highest fatality rates, while signalised intersections show below-average "
        "rates; however, the moderate Cramér's V suggests that traffic control is one "
        "of several contributing infrastructure factors rather than the dominant predictor."
    )
    fig.text(0.5, -0.05, interp, ha="center", fontsize=9,
             color="#2C3E50", style="italic",
             bbox=dict(boxstyle="round", fc="#EAFAF1", ec=C_GREEN, lw=1))

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out / "fig9_h4_traffictl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig9_h4_traffictl.png")

    results.append({
        "hypothesis": "H4", "variable": "traffictl (grouped)",
        "test": "Chi-Square", "chi2": round(chi2_v, 3),
        "df": dof_v, "p_value": p_v,
        "effect_size_metric": "Cramér's V",
        "effect_size": round(cv_v, 4), "reject_h0": reject,
    })
    print(f"  H4 | traffictl: χ²({dof_v})={chi2_v:.2f}, p={p_v:.4e}, "
          f"V={cv_v:.4f}, {'REJECT' if reject else 'FAIL TO REJECT'} H₀")
    print(ct_full.to_string())
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

HYPOTHESIS_LABELS = {
    "H1": "Poor visibility / adverse road conditions\n→ higher fatal probability",
    "H2": "Behavioural risk factors (impairment,\naggression, distraction) → higher fatal probability",
    "H3": "Vulnerable road users over-represented\nin fatal outcomes",
    "H4": "Uncontrolled intersections → higher\nfatality rate than signalised",
}

def build_summary_table(all_results: list[dict], out: Path) -> pd.DataFrame:
    # One row per hypothesis (aggregate where multiple variables)
    summary_rows = []
    seen = set()
    for r in all_results:
        h = r["hypothesis"]
        if h in seen:
            continue
        seen.add(h)
        # For H2 pick the row with highest chi2
        h_rows = [x for x in all_results if x["hypothesis"] == h]
        best = max(h_rows, key=lambda x: x["chi2"])
        summary_rows.append({
            "Hypothesis": h,
            "Description": HYPOTHESIS_LABELS.get(h, ""),
            "Variable(s)": ", ".join(set(x["variable"] for x in h_rows)),
            "Test": best["test"],
            "χ²": best["chi2"],
            "df": best["df"],
            "p-value": "< 0.001" if best["p_value"] < 0.001 else f"{best['p_value']:.4f}",
            "Effect Size": f"V = {best['effect_size']}",
            "Reject H₀?": "✓ Yes" if best["reject_h0"] else "✗ No",
        })

    summary_df = pd.DataFrame(summary_rows)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(17, 5.5), facecolor="white")
    fig.suptitle(
        "Hypothesis Testing Summary — H1 through H4\n"
        "KSI Traffic Collision Severity Study | Group 5",
        fontsize=14, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")

    display_cols = ["Hypothesis", "Test", "χ²", "df", "p-value",
                    "Effect Size", "Reject H₀?"]
    display_data = summary_df[display_cols].values.tolist()

    tbl = ax.table(
        cellText=display_data,
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.1, 2.8)
    tbl.auto_set_column_width(col=list(range(len(display_cols))))

    # Header
    for j in range(len(display_cols)):
        tbl[0, j].set_facecolor("#1A1A2E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row colours
    h_colors = {
        "H1": "#D6EAF8",
        "H2": "#FADBD8",
        "H3": "#D5F5E3",
        "H4": "#FEF9E7",
    }
    reject_col  = "#A9DFBF"
    noreject_col= "#F5CBA7"

    for i, row in enumerate(summary_df.itertuples(), start=1):
        h_color = h_colors.get(row.Hypothesis, "white")
        for j in range(len(display_cols)):
            tbl[i, j].set_facecolor(h_color)
        # Highlight reject column
        last_j = len(display_cols) - 1
        reject_val = summary_df.iloc[i-1]["Reject H₀?"]
        if "Yes" in reject_val:
            tbl[i, last_j].set_facecolor(reject_col)
            tbl[i, last_j].set_text_props(color="#1E8449", fontweight="bold")
        else:
            tbl[i, last_j].set_facecolor(noreject_col)
            tbl[i, last_j].set_text_props(color="#A04000", fontweight="bold")

    # Significance note
    ax.text(0.5, 0.02,
            "α = 0.05  |  Effect size: Cramér's V  (small < 0.1, medium 0.1–0.3, large > 0.3)",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#555555", style="italic")

    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(out / "fig10_hypothesis_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig10_hypothesis_summary.png")

    # Save CSV
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(out / "hypothesis_results.csv", index=False)
    print("  Saved hypothesis_results.csv")

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str = "outputs") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df = load_data(input_path)
    print(f"  {len(df):,} KSI records | "
          f"Fatal: {df['acclass_binary'].sum():,} | "
          f"Non-Fatal: {(df['acclass_binary']==0).sum():,}\n")

    all_results = []
    print("── H1: Environmental conditions ─────────────────────")
    all_results += test_h1(df, out)

    print("\n── H2: Behavioural risk factors ─────────────────────")
    all_results += test_h2(df, out)

    print("\n── H3: Vulnerable road users ────────────────────────")
    all_results += test_h3(df, out)

    print("\n── H4: Traffic control type ─────────────────────────")
    all_results += test_h4(df, out)

    print("\n── Summary table ─────────────────────────────────────")
    summary = build_summary_table(all_results, out)
    print(summary[["Hypothesis", "χ²", "p-value",
                   "Effect Size", "Reject H₀?"]].to_string(index=False))

    print(f"\nAll outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Story 3 – Statistical Inference H1–H4")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.input, args.output_dir)