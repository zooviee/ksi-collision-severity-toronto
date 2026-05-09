import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
# =========================================================
# STORY 2 - EDA VISUALIZATIONS
# Toronto KSI Collision Severity Project
#
# This script performs:
# Task #12 - Severity distribution analysis
# Task #13 - Categorical predictor distributions
# Task #14 - Temporal trend visualizations
# Task #15 - Summary statistics generation
# Task #16 - Fatal vs non-fatal proportion analysis
# =========================================================


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fatal_label(series):
    return series.map({1: "Fatal", 0: "Non-Fatal"}).fillna(series)


def save_fig(output_dir, filename):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# TASK #12
# Plot distribution of acclass (fatal vs non-fatal)
# as a bar chart with percentage labels.
# ---------------------------------------------------------
def plot_acclass_distribution(df, output_dir):
    counts = df["acclass_binary"].value_counts().sort_index()
    labels = ["Non-Fatal" if i == 0 else "Fatal" for i in counts.index]
    percentages = counts / counts.sum() * 100

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, counts.values)

    for bar, pct in zip(bars, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title("Distribution of Collision Severity: Fatal vs Non-Fatal")
    plt.xlabel("Collision Severity")
    plt.ylabel("Number of Records")

    save_fig(output_dir, "task_12_acclass_distribution.png")


def get_onehot_distribution(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix + "_")]

    if not cols:
        return None

    counts = df[cols].sum().sort_values(ascending=True)
    counts.index = [col.replace(prefix + "_", "") for col in counts.index]

    return counts


# ---------------------------------------------------------
# TASK #13
# Plot frequency distributions for all categorical predictors:
# light, rdsfcond, traffictl, road_class, accloc.
# ---------------------------------------------------------
def plot_categorical_distributions(df, output_dir):
    categorical_prefixes = ["light", "rdsfcond", "traffictl", "road_class", "accloc"]

    for prefix in categorical_prefixes:
        counts = get_onehot_distribution(df, prefix)

        if counts is None or counts.empty:
            print(f"Skipping {prefix}: no matching one-hot columns found.")
            continue

        counts = counts.sort_values(ascending=True)

        plt.figure(figsize=(10, max(5, len(counts) * 0.35)))
        plt.barh(counts.index, counts.values)

        plt.title(f"Frequency Distribution of {prefix}")
        plt.xlabel("Frequency")
        plt.ylabel(prefix)

        save_fig(output_dir, f"task_13_{prefix}_distribution.png")


# ---------------------------------------------------------
# TASK #14
# Plot temporal distributions:
# collisions by hour, day_of_week, month, and year.
# ---------------------------------------------------------
def plot_temporal_distributions(df, output_dir):
    temporal_columns = {
        "hour": "Collisions by Hour",
        "day_of_week_name": "Collisions by Day of Week",
        "month_name": "Collisions by Month",
        "year": "Collisions by Year",
    }

    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    for col, title in temporal_columns.items():
        if col not in df.columns:
            print(f"Skipping {col}: column not found.")
            continue

        counts = df[col].value_counts()

        if col == "hour":
            counts = counts.reindex(range(24), fill_value=0)
        elif col == "day_of_week_name":
            counts = counts.reindex(day_order, fill_value=0)
        elif col == "month_name":
            counts = counts.reindex(month_order, fill_value=0)
        else:
            counts = counts.sort_index()

        plt.figure(figsize=(10, 5))
        bars = plt.bar(counts.index.astype(str), counts.values)

        peak_index = counts.values.argmax()
        peak_label = counts.index[peak_index]
        peak_value = counts.values[peak_index]

        plt.text(
            peak_index,
            peak_value,
            f"Peak: {peak_label}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Number of Collisions")
        plt.xticks(rotation=45)

        save_fig(output_dir, f"task_14_{col}_temporal_distribution.png")


# ---------------------------------------------------------
# TASK #15
# Produce summary statistics table for numeric variable invage:
# mean, median, std, min, max, and quartiles.
# Group statistics by Fatal vs Non-Fatal.
# ---------------------------------------------------------
def create_summary_statistics(df, output_dir):
    if "invage" not in df.columns:
        print("Skipping summary statistics: invage column not found.")
        return

    if "acclass_binary" not in df.columns:
        print("Skipping summary statistics: acclass_binary column not found.")
        return

    df_copy = df.copy()
    df_copy["severity_group"] = fatal_label(df_copy["acclass_binary"])

    summary = df_copy.groupby("severity_group")["invage"].describe()
    summary["median"] = df_copy.groupby("severity_group")["invage"].median()

    summary = summary[
        ["mean", "median", "std", "min", "25%", "50%", "75%", "max"]
    ]

    summary.to_csv(output_dir / "task_15_invage_summary_statistics_by_severity.csv")

    print("Saved Task #15 summary statistics table.")


# ---------------------------------------------------------
# TASK #16
# Plot stacked bar charts showing proportions of fatal vs
# non-fatal within each category:
# light, rdsfcond, traffictl, road_user.
# ---------------------------------------------------------
def plot_stacked_proportions(df, output_dir):
    onehot_prefixes = ["light", "rdsfcond", "traffictl"]

    for prefix in onehot_prefixes:
        onehot_cols = [col for col in df.columns if col.startswith(prefix + "_")]

        if not onehot_cols:
            print(f"Skipping stacked chart for {prefix}: no matching one-hot columns found.")
            continue

        rows = []

        for col in onehot_cols:
            category = col.replace(prefix + "_", "")
            subset = df[df[col] == 1]

            if subset.empty:
                continue

            severity_counts = subset["acclass_binary"].value_counts(normalize=True) * 100

            rows.append({
                "category": category,
                "Non-Fatal": severity_counts.get(0, 0),
                "Fatal": severity_counts.get(1, 0),
            })

        prop_df = pd.DataFrame(rows)

        if prop_df.empty:
            print(f"Skipping stacked chart for {prefix}: no valid category data.")
            continue

        prop_df = prop_df.set_index("category").sort_values("Fatal", ascending=True)

        ax = prop_df[["Non-Fatal", "Fatal"]].plot(
            kind="barh",
            stacked=True,
            figsize=(10, max(5, len(prop_df) * 0.35)),
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=8)

        plt.title(f"Fatal vs Non-Fatal Proportions by {prefix}")
        plt.xlabel("Percentage")
        plt.ylabel(prefix)

        save_fig(output_dir, f"task_16_{prefix}_fatal_nonfatal_stacked.png")

    if "road_user" not in df.columns:
        print("Skipping stacked chart for road_user: column not found.")
        return

    table = pd.crosstab(
        df["road_user"],
        df["acclass_binary"],
        normalize="index"
    ) * 100

    table = table.rename(columns={0: "Non-Fatal", 1: "Fatal"})

    for needed_col in ["Non-Fatal", "Fatal"]:
        if needed_col not in table.columns:
            table[needed_col] = 0

    table = table[["Non-Fatal", "Fatal"]].sort_values("Fatal", ascending=True)

    ax = table.plot(
        kind="barh",
        stacked=True,
        figsize=(10, max(5, len(table) * 0.35)),
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=8)

    plt.title("Fatal vs Non-Fatal Proportions by road_user")
    plt.xlabel("Percentage")
    plt.ylabel("road_user")

    save_fig(output_dir, "task_16_road_user_fatal_nonfatal_stacked.png")


def run_eda(input_path, output_dir):
    output_dir = ensure_output_dir(output_dir)

    df = pd.read_csv(input_path)

    if "acclass_binary" not in df.columns:
        raise ValueError("Required column missing: acclass_binary")

    plot_acclass_distribution(df, output_dir)
    plot_categorical_distributions(df, output_dir)
    plot_temporal_distributions(df, output_dir)
    create_summary_statistics(df, output_dir)
    plot_stacked_proportions(df, output_dir)

    print("Story 2 EDA visualizations complete.")
    print(f"Outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=r"C:\Users\martf\Downloads\Capstone_Project\ksi-collision-severity-toronto\outputs\story-1\ksi_encoded.csv",
        help="Path to encoded KSI dataset from Story 1",
    )

    parser.add_argument(
        "--output",
        default="outputs/story-2",
        help="Folder to save Story 2 EDA outputs",
    )

    args = parser.parse_args()
    run_eda(args.input, args.output)


if __name__ == "__main__":
    main()