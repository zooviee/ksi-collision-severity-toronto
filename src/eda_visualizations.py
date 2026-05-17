"""
Story 2 – EDA Visualizations
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5

Tasks
─────
Task #12 - Severity distribution analysis (acclass_binary)
Task #13 - Categorical predictor distributions (one-hot encoded columns)
Task #14 - Temporal trend visualizations (hour, day, month, year)
Task #15 - Summary statistics by severity group (invage)
Task #16 - Fatal vs non-fatal stacked proportion charts

Input:
    outputs/story-1/ksi_encoded.csv   ← produced by Story 1

Output:
    outputs/story-2/
        task_12_acclass_distribution.png
        task_13_<prefix>_distribution.png   (×5)
        task_14_<col>_temporal_distribution.png  (×4)
        task_15_invage_summary_statistics_by_severity.csv
        task_16_<prefix>_fatal_nonfatal_stacked.png  (×4)

Usage:
    python src/eda_visualizations.py \\
        --input  outputs/story-1/ksi_encoded.csv \\
        --output-dir outputs/story-2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Colour palette ─────────────────────────────────────────────────────────
C_FATAL    = "#C0392B"
C_NONFATAL = "#2980B9"
C_ACCENT   = "#E67E22"
C_GREEN    = "#27AE60"
C_BG       = "#F8F9FA"
C_GRID     = "#DEE2E6"

FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1A1A2E"}
FONT_AX    = {"fontsize": 10, "color": "#2C3E50"}

# ── Shared axis styler ──────────────────────────────────────────────────────
def _style_ax(ax, xlabel="", ylabel="", title="", grid_axis="y"):
    ax.set_facecolor(C_BG)
    ax.set_title(title, **FONT_TITLE, pad=8)
    ax.set_xlabel(xlabel, **FONT_AX)
    ax.set_ylabel(ylabel, **FONT_AX)
    ax.tick_params(labelsize=9, colors="#4A4A4A")
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)
    ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


# =========================================================
# Helpers
# =========================================================

def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fatal_label(series):
    return series.map({1: "Fatal", 0: "Non-Fatal"}).fillna(series)


def save_fig(output_dir, filename):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def get_onehot_distribution(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix + "_")]
    if not cols:
        return None
    counts = df[cols].sum().sort_values(ascending=True)
    counts.index = [col.replace(prefix + "_", "") for col in counts.index]
    return counts


# =========================================================
# TASK #12 — Severity distribution
# =========================================================

def plot_acclass_distribution(df, output_dir):
    """
    Bar chart of fatal vs non-fatal collision counts with
    percentage labels and colour coding.
    """
    counts      = df["acclass_binary"].value_counts().sort_index()
    labels      = ["Non-Fatal" if i == 0 else "Fatal" for i in counts.index]
    percentages = counts / counts.sum() * 100
    colors      = [C_NONFATAL, C_FATAL]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    bars = ax.bar(labels, counts.values, color=colors,
                  edgecolor="white", width=0.5, zorder=3)

    for bar, pct, cnt in zip(bars, percentages, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"{pct:.1f}%\n({cnt:,})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    _style_ax(ax,
              xlabel="Collision Severity",
              ylabel="Number of Records",
              title="Fig 12 — Distribution of Collision Severity\n"
                    "Fatal vs Non-Fatal KSI Records (2006–2026)",
              grid_axis="y")

    # Imbalance annotation
    ratio = counts.get(0, 0) / counts.get(1, 1)
    ax.text(0.98, 0.96,
            f"Imbalance ratio\n{ratio:.1f}:1 (Non-Fatal:Fatal)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=C_ACCENT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=C_ACCENT, lw=1.2))

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "task_12_acclass_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_12_acclass_distribution.png")


# =========================================================
# TASK #13 — Categorical predictor distributions
# =========================================================

def plot_categorical_distributions(df, output_dir):
    """
    Horizontal bar charts for each one-hot encoded categorical predictor.
    Bars are coloured by frequency relative to the mean.
    """
    categorical_prefixes = ["light", "rdsfcond", "traffictl", "road_class", "accloc"]

    for prefix in categorical_prefixes:
        counts = get_onehot_distribution(df, prefix)

        if counts is None or counts.empty:
            print(f"  Skipping {prefix}: no matching one-hot columns found.")
            continue

        counts = counts.sort_values(ascending=True)
        mean_count = counts.mean()
        colors = [C_FATAL if v >= mean_count else C_NONFATAL for v in counts.values]

        fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.5)),
                               facecolor="white")
        bars = ax.barh(counts.index, counts.values,
                       color=colors, edgecolor="white", zorder=3)

        for bar, v in zip(bars, counts.values):
            ax.text(v + counts.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(v):,}", va="center", fontsize=8.5)

        ax.axvline(mean_count, color=C_ACCENT, linestyle="--",
                   linewidth=1.5, label=f"Mean ({int(mean_count):,})")

        _style_ax(ax,
                  xlabel="Frequency (record count)",
                  ylabel=prefix,
                  title=f"Fig 13 — Frequency Distribution: {prefix}\n"
                        f"KSI Records 2006–2026 · red = above average",
                  grid_axis="x")

        legend_patches = [
            mpatches.Patch(color=C_FATAL,    label="Above average"),
            mpatches.Patch(color=C_NONFATAL, label="Below average"),
            plt.Line2D([0], [0], color=C_ACCENT, linestyle="--",
                       lw=1.5, label=f"Mean ({int(mean_count):,})"),
        ]
        ax.legend(handles=legend_patches, fontsize=8.5, framealpha=0.9)

        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"task_13_{prefix}_distribution.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved task_13_{prefix}_distribution.png")


# =========================================================
# TASK #14 — Temporal distributions
# =========================================================

def plot_temporal_distributions(df, output_dir):
    """
    Bar charts for collisions by hour, day of week, month, and year.
    Bars are coloured by volume; peak is annotated; year chart
    includes COVID dip and data-lag annotations.
    """
    temporal_columns = {
        "hour":            "Collisions by Hour of Day",
        "day_of_week_name":"Collisions by Day of Week",
        "month_name":      "Collisions by Month",
        "year":            "Collisions by Year (2006–2023)",
    }

    day_order = ["Monday","Tuesday","Wednesday","Thursday",
                 "Friday","Saturday","Sunday"]
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]

    LAG_YEARS = [2024, 2025, 2026]

    for col, title in temporal_columns.items():
        if col not in df.columns:
            print(f"  Skipping {col}: column not found.")
            continue

        counts = df[col].value_counts()

        if col == "hour":
            counts = counts.reindex(range(24), fill_value=0)
        elif col == "day_of_week_name":
            counts = counts.reindex(day_order, fill_value=0)
        elif col == "month_name":
            counts = counts.reindex(month_order, fill_value=0)
        elif col == "year":
            counts = counts.sort_index()
            counts = counts[counts.index <= 2023]   # exclude lag years from trend
        else:
            counts = counts.sort_index()

        mean_val = counts.mean()

        # Colour bars by volume
        if col == "hour":
            bar_colors = [C_FATAL if v >= mean_val * 1.1
                          else C_ACCENT if v >= mean_val
                          else C_NONFATAL for v in counts.values]
        else:
            bar_colors = [C_FATAL if v >= mean_val else C_NONFATAL
                          for v in counts.values]

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
        bars = ax.bar(counts.index.astype(str), counts.values,
                      color=bar_colors, edgecolor="white", zorder=3)

        # Peak annotation
        peak_idx   = counts.values.argmax()
        peak_label = counts.index[peak_idx]
        peak_value = counts.values[peak_idx]
        ax.annotate(
            f"Peak: {peak_label}\n({int(peak_value):,})",
            xy=(peak_idx, peak_value),
            xytext=(peak_idx + max(1, len(counts) * 0.07), peak_value * 0.95),
            fontsize=9, fontweight="bold", color=C_FATAL,
            arrowprops=dict(arrowstyle="->", color=C_FATAL, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FATAL, lw=1),
        )

        # Mean reference line
        ax.axhline(mean_val, color=C_ACCENT, linestyle="--",
                   linewidth=1.5, label=f"Mean ({int(mean_val):,})")

        # COVID annotation on year chart
        if col == "year":
            year_labels = [str(y) for y in counts.index]
            if "2020" in year_labels:
                covid_pos = year_labels.index("2020")
                ax.annotate(
                    "COVID-19\ndip",
                    xy=(covid_pos, counts.loc[2020]),
                    xytext=(covid_pos + 0.8, counts.loc[2020] + 50),
                    fontsize=8, color=C_ACCENT,
                    arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1),
                )
            ax.text(0.98, 0.96,
                    "⚠ 2024–2026 excluded\n(police reporting lag)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=C_ACCENT,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=C_ACCENT, lw=1))

        _style_ax(ax,
                  xlabel=col.replace("_", " ").title(),
                  ylabel="Number of Collisions",
                  title=f"Fig 14 — {title}",
                  grid_axis="y")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8.5, framealpha=0.9)

        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"task_14_{col}_temporal_distribution.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved task_14_{col}_temporal_distribution.png")


# =========================================================
# TASK #15 — Summary statistics
# =========================================================

def create_summary_statistics(df, output_dir):
    """
    Summary statistics table for invage grouped by Fatal / Non-Fatal.
    Saved as CSV and as a styled PNG figure.
    """
    if "invage" not in df.columns:
        print("  Skipping summary statistics: invage column not found.")
        return
    if "acclass_binary" not in df.columns:
        print("  Skipping summary statistics: acclass_binary column not found.")
        return

    df_copy = df.copy()
    df_copy["severity_group"] = fatal_label(df_copy["acclass_binary"])

    summary = df_copy.groupby("severity_group")["invage"].describe()
    summary["median"] = df_copy.groupby("severity_group")["invage"].median()
    summary = summary[["mean","median","std","min","25%","50%","75%","max"]]
    summary = summary.round(2)

    # Save CSV
    csv_path = Path(output_dir) / "task_15_invage_summary_statistics_by_severity.csv"
    summary.to_csv(csv_path)
    print("  Saved task_15_invage_summary_statistics_by_severity.csv")

    # Save as styled PNG table
    fig, ax = plt.subplots(figsize=(13, 3), facecolor="white")
    fig.suptitle(
        "Fig 15 — Age of Involved Person (invage): Summary Statistics by Severity\n"
        "Fatal vs Non-Fatal KSI Collisions · 2006–2026",
        **FONT_TITLE
    )
    ax.axis("off")

    tbl = ax.table(
        cellText=summary.reset_index().values.tolist(),
        colLabels=["Severity"] + list(summary.columns),
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.4)

    for j in range(len(summary.columns) + 1):
        tbl[0, j].set_facecolor("#1A1A2E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fills = {"Fatal": "#FADBD8", "Non-Fatal": "#D6EAF8"}
    for i, row_label in enumerate(summary.index, start=1):
        fill = fills.get(row_label, "white")
        for j in range(len(summary.columns) + 1):
            tbl[i, j].set_facecolor(fill)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(Path(output_dir) / "task_15_invage_summary_table.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_15_invage_summary_table.png")


# =========================================================
# TASK #16 — Stacked proportion charts
# =========================================================

def plot_stacked_proportions(df, output_dir):
    """
    Horizontal stacked bar charts showing % fatal vs non-fatal
    within each category. Bars sorted by fatal rate descending.
    Reference line shows overall average fatal rate.
    """
    overall_fatal_rate = df["acclass_binary"].mean() * 100

    # ── One-hot encoded prefixes ──────────────────────────────
    onehot_prefixes = ["light", "rdsfcond", "traffictl"]

    for prefix in onehot_prefixes:
        onehot_cols = [c for c in df.columns if c.startswith(prefix + "_")]

        if not onehot_cols:
            print(f"  Skipping stacked chart for {prefix}: no OHE columns found.")
            continue

        rows = []
        for col in onehot_cols:
            category = col.replace(prefix + "_", "")
            subset = df[df[col] == 1]
            if subset.empty:
                continue
            sev = subset["acclass_binary"].value_counts(normalize=True) * 100
            rows.append({
                "category":  category,
                "Non-Fatal": sev.get(0, 0),
                "Fatal":     sev.get(1, 0),
            })

        if not rows:
            continue

        prop_df = (pd.DataFrame(rows)
                   .set_index("category")
                   .sort_values("Fatal", ascending=True))

        fig, ax = plt.subplots(figsize=(11, max(5, len(prop_df) * 0.55)),
                               facecolor="white")

        prop_df[["Non-Fatal", "Fatal"]].plot(
            kind="barh", stacked=True, ax=ax,
            color=[C_NONFATAL, C_FATAL], edgecolor="white"
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%",
                         label_type="center", fontsize=8.5,
                         color="white", fontweight="bold")

        # Overall average reference line
        ax.axvline(100 - overall_fatal_rate, color=C_ACCENT,
                   linestyle="--", linewidth=1.5,
                   label=f"Overall Non-Fatal avg ({100-overall_fatal_rate:.1f}%)")

        _style_ax(ax,
                  xlabel="Proportion (%)",
                  ylabel=prefix.replace("_", " ").title(),
                  title=f"Fig 16 — Fatal vs Non-Fatal Proportions by {prefix}\n"
                        f"Sorted by fatal rate · overall fatal rate = {overall_fatal_rate:.1f}%",
                  grid_axis="x")

        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.set_xlim(0, 100)

        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"task_16_{prefix}_fatal_nonfatal_stacked.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved task_16_{prefix}_fatal_nonfatal_stacked.png")

    # ── road_user (raw column, not OHE) ──────────────────────
    if "road_user" not in df.columns:
        print("  Skipping stacked chart for road_user: column not found.")
        return

    table = (pd.crosstab(df["road_user"], df["acclass_binary"], normalize="index") * 100)
    table = table.rename(columns={0: "Non-Fatal", 1: "Fatal"})
    for col in ["Non-Fatal", "Fatal"]:
        if col not in table.columns:
            table[col] = 0
    table = table[["Non-Fatal", "Fatal"]].sort_values("Fatal", ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(5, len(table) * 0.55)),
                           facecolor="white")

    table.plot(kind="barh", stacked=True, ax=ax,
               color=[C_NONFATAL, C_FATAL], edgecolor="white")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%",
                     label_type="center", fontsize=8.5,
                     color="white", fontweight="bold")

    ax.axvline(100 - overall_fatal_rate, color=C_ACCENT,
               linestyle="--", linewidth=1.5,
               label=f"Overall Non-Fatal avg ({100-overall_fatal_rate:.1f}%)")

    _style_ax(ax,
              xlabel="Proportion (%)",
              ylabel="Road User Type",
              title=f"Fig 16 — Fatal vs Non-Fatal Proportions by Road User\n"
                    f"Sorted by fatal rate · overall fatal rate = {overall_fatal_rate:.1f}%",
              grid_axis="x")

    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "task_16_road_user_fatal_nonfatal_stacked.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_16_road_user_fatal_nonfatal_stacked.png")


# =========================================================
# Orchestrator
# =========================================================

def run_eda(input_path, output_dir):
    output_dir = ensure_output_dir(output_dir)

    df = pd.read_csv(input_path)

    if "acclass_binary" not in df.columns:
        raise ValueError(
            "Required column 'acclass_binary' not found.\n"
            "Run Story 1 first: python src/data_preparation.py --input ... --output-dir outputs/story-1\n"
            "Then pass --input outputs/story-1/ksi_encoded.csv"
        )

    print(f"  Loaded {len(df):,} records · {df.shape[1]} columns")
    print(f"  Fatal: {df['acclass_binary'].sum():,}  "
          f"Non-Fatal: {(df['acclass_binary']==0).sum():,}")

    plot_acclass_distribution(df, output_dir)
    plot_categorical_distributions(df, output_dir)
    plot_temporal_distributions(df, output_dir)
    create_summary_statistics(df, output_dir)
    plot_stacked_proportions(df, output_dir)

    print(f"\n  Story 2 EDA complete — outputs saved to {output_dir}")


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Story 2 – EDA Visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/eda_visualizations.py \\
      --input      outputs/story-1/ksi_encoded.csv \\
      --output-dir outputs/story-2
        """
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to ksi_encoded.csv from Story 1 (outputs/story-1/ksi_encoded.csv)"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/story-2",
        help="Directory to save Story 2 outputs (default: outputs/story-2)"
    )

    args = parser.parse_args()
    run_eda(args.input, args.output_dir)


if __name__ == "__main__":
    main()