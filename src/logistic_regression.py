"""
Story 4: Quantify Predictor Effects via Binary Logistic Regression

Tasks:
#24 Fit binary logistic regression using key predictors on imbalanced data.
#25 Extract coefficient table: Predictor, Coef, Std Err, z, p-value, OR, 95% CI.
#26 Check multicollinearity using VIF and resolve VIF > 5.
#27 Interpret top 5 significant predictors in plain language.

Input:
    outputs/story-1/ksi_encoded.csv

Output:
    outputs/story-4/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================================================
# Load data
# =========================================================

def load_data(input_path):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    if "acclass_binary" not in df.columns:
        raise ValueError("acclass_binary not found. Run Story 1 first.")

    print("\nTarget distribution:")
    print(df["acclass_binary"].value_counts())

    return df


# =========================================================
# Task #24: Prepare predictors
# =========================================================

def drop_one_reference_category(X, prefix):
    """
    Drop one one-hot encoded category per variable group
    to avoid perfect multicollinearity.
    """
    cols = [col for col in X.columns if col.startswith(prefix + "_")]

    if len(cols) > 1:
        reference_col = sorted(cols)[0]
        X = X.drop(columns=[reference_col])
        print(f"Dropped reference category for {prefix}: {reference_col}")

    return X


def prepare_predictors(df):
    """
    Prepare key predictors for logistic regression.

    Important:
    - We remove injury and injury_encoded because they are leakage.
    - We remove acclass/fatal columns because they directly reveal the target.
    - We use Story 1 encoded predictors.
    """

    y = df["acclass_binary"].astype(int)

    # Keep numeric columns only because Story 1 already encoded the main categorical predictors.
    X = df.select_dtypes(include=[np.number]).copy()

    # Remove direct target and leakage columns.
    leakage_keywords = [
        "acclass",
        "fatal",
        "injury",
    ]

    leakage_cols = [
        col for col in X.columns
        if any(keyword in col.lower() for keyword in leakage_keywords)
    ]

    # Remove ID-style columns.
    id_cols = [
        "_id",
        "collision_id",
    ]

    X = X.drop(columns=leakage_cols + id_cols, errors="ignore")

    # Keep key predictor families only.
    key_base_predictors = [
        "per_inv",
        "invage",
        "aggressive",
        "distracted",
        "pedestrian",
        "cyclist",
        "motorcyclist",
        "red_light",
        "older_adult",
        "school_child",
        "heavy_truck",
        "drivcond_missing",
        "aggressive_missing",
        "distracted_missing",
        "year",
        "month",
        "hour",
        "is_weekend",
    ]

    key_prefixes = [
        "light_",
        "rdsfcond_",
        "traffictl_",
        "road_class_",
        "accloc_",
        "impactype_",
    ]

    keep_cols = []

    for col in key_base_predictors:
        if col in X.columns:
            keep_cols.append(col)

    for col in X.columns:
        if any(col.startswith(prefix) for prefix in key_prefixes):
            keep_cols.append(col)

    keep_cols = sorted(set(keep_cols))

    X = X[keep_cols].copy()

    # Drop one reference category per one-hot group.
    for prefix in ["light", "rdsfcond", "traffictl", "road_class", "accloc", "impactype"]:
        X = drop_one_reference_category(X, prefix)

    # Clean missing or infinite values.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Remove columns with no variation.
    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=True) <= 1
    ]

    X = X.drop(columns=constant_cols, errors="ignore")

    print("\nTask #24 predictor preparation complete.")
    print(f"Predictors before VIF check: {X.shape[1]}")

    print("\nPredictor columns used:")
    for col in X.columns:
        print(f"- {col}")

    return X, y


# =========================================================
# Task #26: VIF
# =========================================================

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor for each predictor.
    """
    X_const = sm.add_constant(X, has_constant="add")

    rows = []

    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue

        try:
            vif = variance_inflation_factor(X_const.values, i)
        except Exception:
            vif = np.inf

        rows.append({
            "Predictor": col,
            "VIF": vif,
        })

    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def reduce_multicollinearity(X, threshold=5.0):
    """
    Drop predictors one by one until all VIF values are <= threshold.
    """
    X_reduced = X.copy()
    dropped = []

    while True:
        vif_df = calculate_vif(X_reduced)

        if vif_df.empty:
            break

        max_vif = vif_df.iloc[0]["VIF"]
        max_col = vif_df.iloc[0]["Predictor"]

        if max_vif <= threshold:
            break

        print(f"Dropping high-VIF predictor: {max_col} | VIF={max_vif:.2f}")

        dropped.append({
            "Dropped Predictor": max_col,
            "VIF": max_vif,
            "Reason": f"VIF greater than {threshold}",
        })

        X_reduced = X_reduced.drop(columns=[max_col])

        if X_reduced.shape[1] <= 1:
            break

    final_vif = calculate_vif(X_reduced)
    dropped_vif = pd.DataFrame(dropped)

    print("\nTask #26 complete: VIF check finished.")
    print(f"Predictors after VIF check: {X_reduced.shape[1]}")

    return X_reduced, final_vif, dropped_vif


# =========================================================
# Task #24: Fit model
# =========================================================

def fit_logistic_model(X, y):
    """
    Fit binary logistic regression using GLM Binomial.

    This is logistic regression, but it is more stable than statsmodels Logit
    for this dataset.
    """

    X_const = sm.add_constant(X, has_constant="add")

    model = sm.GLM(y, X_const, family=sm.families.Binomial())
    result = model.fit(maxiter=300)

    print("\nTask #24 complete: Binary logistic regression fitted.")

    return result


# =========================================================
# Task #25: Coefficient table
# =========================================================

def safe_exp(values):
    """
    Safely exponentiate values for odds ratios and confidence intervals.

    Clipping prevents overflow warnings when coefficients or confidence
    interval bounds are extremely large.
    """
    return np.exp(np.clip(values, -20, 20))


def build_coefficient_table(result):
    """
    Build coefficient table:
    Predictor, Coefficient, Std Error, z, p-value, OR, 95% CI.
    """
    params = result.params
    conf = result.conf_int()

    coef_table = pd.DataFrame({
        "Predictor": params.index,
        "Coefficient": params.values,
        "Std Error": result.bse.values,
        "z": result.tvalues.values,
        "p-value": result.pvalues.values,
        "OR": safe_exp(params.values),
        "95% CI Lower": safe_exp(conf[0].values),
        "95% CI Upper": safe_exp(conf[1].values),
    })

    coef_table = coef_table[coef_table["Predictor"] != "const"].copy()
    coef_table["Significant p<0.05"] = coef_table["p-value"] < 0.05
    coef_table = coef_table.sort_values("p-value", ascending=True)

    print("\nTask #25 complete: Coefficient table created.")

    return coef_table


# =========================================================
# Task #27: Interpretation
# =========================================================

def interpret_top_predictors(coef_table):
    """
    Interpret top 5 statistically significant predictors in plain language.
    """
    significant = coef_table[
        (coef_table["Significant p<0.05"] == True)
        & (coef_table["OR"].replace([np.inf, -np.inf], np.nan).notna())
    ].copy()

    significant = significant.sort_values("p-value").head(5)

    lines = []
    lines.append("Task #27: Top 5 Significant Predictors Interpretation")
    lines.append("=" * 60)
    lines.append("")

    if significant.empty:
        lines.append(
            "No predictors were statistically significant at p < 0.05 "
            "after VIF-based multicollinearity handling."
        )
        return "\n".join(lines)

    for i, (_, row) in enumerate(significant.iterrows(), start=1):
        predictor = row["Predictor"]
        odds_ratio = row["OR"]
        lower_ci = row["95% CI Lower"]
        upper_ci = row["95% CI Upper"]
        p_value = row["p-value"]

        if odds_ratio > 1:
            text = (
                f"{i}. {predictor}: This predictor is associated with higher odds "
                f"of a fatal collision. Holding other predictors constant, the odds "
                f"of a fatal collision are {odds_ratio:.2f} times higher compared "
                f"with the reference category or baseline level "
                f"(OR={odds_ratio:.2f}, 95% CI={lower_ci:.2f}-{upper_ci:.2f}, "
                f"p={p_value:.4f})."
            )
        else:
            reduction = (1 - odds_ratio) * 100
            text = (
                f"{i}. {predictor}: This predictor is associated with lower odds "
                f"of a fatal collision. Holding other predictors constant, the odds "
                f"of a fatal collision are about {reduction:.1f}% lower compared "
                f"with the reference category or baseline level "
                f"(OR={odds_ratio:.2f}, 95% CI={lower_ci:.2f}-{upper_ci:.2f}, "
                f"p={p_value:.4f})."
            )

        lines.append(text)
        lines.append("")

    return "\n".join(lines)


# =========================================================
# Main
# =========================================================

def main(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Story 1 encoded data
    df = load_data(input_path)

    # Task #24: Prepare predictors
    X, y = prepare_predictors(df)

    # Task #26: Check VIF and reduce multicollinearity
    X_reduced, final_vif, dropped_vif = reduce_multicollinearity(X, threshold=5.0)

    final_vif.to_csv(output_dir / "task_26_final_vif_table.csv", index=False)
    dropped_vif.to_csv(output_dir / "task_26_dropped_high_vif_predictors.csv", index=False)

    # Task #24: Fit logistic regression
    result = fit_logistic_model(X_reduced, y)

    with open(output_dir / "task_24_logistic_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(result.summary()))

    # Task #25: Coefficient table
    coef_table = build_coefficient_table(result)
    coef_table.to_csv(output_dir / "task_25_logistic_regression_coefficients.csv", index=False)

    significant_table = coef_table[coef_table["Significant p<0.05"] == True].copy()
    significant_table.to_csv(output_dir / "task_25_significant_predictors.csv", index=False)

    # Task #27: Interpretation
    interpretation = interpret_top_predictors(coef_table)

    with open(output_dir / "task_27_top_5_predictors_interpretation.txt", "w", encoding="utf-8") as f:
        f.write(interpretation)

    print("\nStory 4 complete.")
    print(f"Outputs saved to: {output_dir}")

    print("\nGenerated files:")
    print("1. task_24_logistic_model_summary.txt")
    print("2. task_25_logistic_regression_coefficients.csv")
    print("3. task_25_significant_predictors.csv")
    print("4. task_26_final_vif_table.csv")
    print("5. task_26_dropped_high_vif_predictors.csv")
    print("6. task_27_top_5_predictors_interpretation.txt")

    print("\nTop 5 significant predictors:")
    print(significant_table.head(5)[[
        "Predictor",
        "Coefficient",
        "Std Error",
        "z",
        "p-value",
        "OR",
        "95% CI Lower",
        "95% CI Upper",
    ]])

    print("\nPlain-language interpretation:")
    print(interpretation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 4: Binary logistic regression for predictor effect interpretation."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to Story 1 encoded dataset, usually outputs/story-1/ksi_encoded.csv"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Story 4 outputs will be saved"
    )

    args = parser.parse_args()

    main(args.input, args.output_dir)