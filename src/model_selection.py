"""
Story 9 – Select Best Model Using Cross Validation

Tasks covered:
#51 Run 10-fold stratified cross-validation on LR, DT, RF, XGBoost.
#52 Produce final model comparison table.
#53 Write model selection rationale.
#54 Save best model as .pkl using joblib and document preprocessing pipeline.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from xgboost import XGBClassifier


RANDOM_STATE = 42

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


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model_data(input_path):
    df = pd.read_csv(input_path, low_memory=False)

    if "acclass_binary" not in df.columns:
        raise ValueError(
            "Column 'acclass_binary' not found. "
            "Use outputs/story-1/ksi_encoded.csv as input."
        )

    available_features = [col for col in CORE_FEATURES if col in df.columns]

    if not available_features:
        raise ValueError("None of the expected model features were found.")

    missing_features = [col for col in CORE_FEATURES if col not in df.columns]
    if missing_features:
        print(f"WARNING: Missing {len(missing_features)} expected features.")
        print(missing_features)

    X = df[available_features].fillna(0)
    y = df["acclass_binary"]

    return X, y, available_features


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=0.01,
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_pipeline(model):
    return Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model),
    ])


# ---------------------------------------------------------
# TASK #51
# Run 10-fold stratified cross-validation on all 4 models:
# Logistic Regression, Decision Tree, Random Forest, XGBoost.
# Report mean ± std for AUC, F1, precision, and recall.
# ---------------------------------------------------------
def run_cross_validation(X, y, n_splits=10):
    models = build_models()

    scoring = {
        "auc": "roc_auc",
        "f1": make_scorer(f1_score, average="macro", zero_division=0),
        "precision": make_scorer(precision_score, average="macro", zero_division=0),
        "recall": make_scorer(recall_score, average="macro", zero_division=0),
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    for model_name, model in models.items():
        print(f"\nRunning {n_splits}-fold CV for {model_name}...")

        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        rows.append({
            "Model": model_name,
            "AUC Mean": scores["test_auc"].mean(),
            "AUC Std": scores["test_auc"].std(),
            "F1 Mean": scores["test_f1"].mean(),
            "F1 Std": scores["test_f1"].std(),
            "Precision Mean": scores["test_precision"].mean(),
            "Precision Std": scores["test_precision"].std(),
            "Recall Mean": scores["test_recall"].mean(),
            "Recall Std": scores["test_recall"].std(),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# TASK #52
# Produce final model comparison table with:
# 4 models x 4 metrics x CV mean and std.
# Format for report and dashboard.
# ---------------------------------------------------------
def save_model_comparison_table(cv_results, output_dir):
    cv_results = cv_results.sort_values(
        by=["AUC Mean", "F1 Mean"],
        ascending=False,
    ).reset_index(drop=True)

    cv_results["Rank"] = cv_results.index + 1

    formatted = cv_results.copy()

    for metric in ["AUC", "F1", "Precision", "Recall"]:
        formatted[f"{metric} CV Mean ± Std"] = (
            formatted[f"{metric} Mean"].map(lambda x: f"{x:.4f}")
            + " ± "
            + formatted[f"{metric} Std"].map(lambda x: f"{x:.4f}")
        )

    final_table = formatted[
        [
            "Rank",
            "Model",
            "AUC CV Mean ± Std",
            "F1 CV Mean ± Std",
            "Precision CV Mean ± Std",
            "Recall CV Mean ± Std",
        ]
    ]

    cv_results.to_csv(output_dir / "task_51_cv_raw_results.csv", index=False)
    final_table.to_csv(output_dir / "task_52_final_model_comparison_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=final_table.values,
        colLabels=final_table.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    plt.title("Final Model Comparison Table – 10-Fold Stratified CV", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "task_52_final_model_comparison_table.png", dpi=300)
    plt.close()

    return cv_results, final_table


# ---------------------------------------------------------
# TASK #53
# Write model selection rationale:
# 2–3 paragraphs explaining why the best model was chosen
# and the trade-off between interpretability and performance.
# ---------------------------------------------------------
def write_model_selection_rationale(cv_results, output_dir):
    best_row = cv_results.iloc[0]
    best_model = best_row["Model"]

    rationale = f"""
Task #53 – Model Selection Rationale

Based on 10-fold stratified cross-validation, the selected best model is {best_model}. This model achieved the strongest overall cross-validated performance, with an AUC mean of {best_row['AUC Mean']:.4f} and F1 mean of {best_row['F1 Mean']:.4f}. AUC was treated as the primary selection metric because the project goal is to distinguish fatal from non-fatal collision outcomes under class imbalance. F1, precision, and recall were also considered to ensure the selected model was not only ranking cases well but also producing useful classification performance.

The main trade-off is between interpretability and predictive performance. Logistic Regression and Decision Tree models are easier to explain because their coefficients, rules, and split logic can be interpreted directly. These models are useful for communicating relationships between collision severity and predictors such as lighting, road surface, traffic control, and road user characteristics. However, simpler models may underfit complex non-linear relationships in traffic collision data.

Random Forest and XGBoost are usually stronger predictive models because they combine many decision rules and can capture interactions between environmental, behavioural, and infrastructural factors. The downside is that they are less transparent than Logistic Regression or a single Decision Tree. For this project, the selected model balances predictive performance with explainability by using cross-validation evidence for selection and relying on feature importance / SHAP-style interpretation outputs from earlier stories to support reporting and dashboard use.
""".strip()

    output_path = output_dir / "task_53_model_selection_rationale.txt"
    output_path.write_text(rationale, encoding="utf-8")

    return rationale


# ---------------------------------------------------------
# TASK #54
# Save best model as .pkl using joblib.
# Document exact preprocessing pipeline required to reproduce
# predictions on new data.
# ---------------------------------------------------------
def save_best_model(cv_results, X, y, feature_columns, output_dir):
    best_model_name = cv_results.iloc[0]["Model"]
    model = build_models()[best_model_name]
    pipeline = build_pipeline(model)

    pipeline.fit(X, y)

    model_path = output_dir / "task_54_best_model.pkl"
    joblib.dump(pipeline, model_path)

    feature_path = output_dir / "task_54_feature_columns.json"
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=4)

    documentation = f"""
Task #54 – Best Model and Reproducibility Documentation

Saved best model:
{model_path}

Selected model:
{best_model_name}

Input dataset expected:
outputs/story-1/ksi_encoded.csv

Required preprocessing pipeline:
1. Start from the raw Toronto KSI collision dataset.
2. Run Story 1 data preparation using src/data_preparation.py.
3. Story 1 must create outputs/story-1/ksi_encoded.csv.
4. The encoded dataset must contain acclass_binary and the same feature columns saved in task_54_feature_columns.json.
5. Missing values in selected numeric/model features are filled with 0.
6. SMOTE is applied only inside the training folds during cross-validation.
7. The saved joblib pipeline includes SMOTE and the final classifier.
8. For new prediction data, the new records must be processed using the same Story 1 transformations and aligned to the saved feature columns before prediction.

Important:
Generated outputs are saved in outputs/story-9 and should not be committed unless the team explicitly requests output artifacts.
""".strip()

    doc_path = output_dir / "task_54_preprocessing_pipeline_documentation.txt"
    doc_path.write_text(documentation, encoding="utf-8")

    return model_path, doc_path


def run_story_9(input_path, output_dir, n_splits):
    output_dir = ensure_output_dir(output_dir)

    print("Loading model data...")
    X, y, feature_columns = load_model_data(input_path)

    print(f"Rows: {len(X):,}")
    print(f"Features used: {len(feature_columns)}")
    print(f"Fatal class rate: {y.mean() * 100:.2f}%")

    cv_results = run_cross_validation(X, y, n_splits=n_splits)

    cv_results, final_table = save_model_comparison_table(cv_results, output_dir)

    rationale = write_model_selection_rationale(cv_results, output_dir)

    model_path, doc_path = save_best_model(
        cv_results,
        X,
        y,
        feature_columns,
        output_dir,
    )

    print("\nStory 9 complete.")
    print("\nFinal comparison table:")
    print(final_table.to_string(index=False))
    print(f"\nBest model saved to: {model_path}")
    print(f"Pipeline documentation saved to: {doc_path}")
    print(f"Outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="outputs/story-1/ksi_encoded.csv",
        help="Path to Story 1 encoded dataset.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/story-9",
        help="Folder to save Story 9 outputs.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of stratified CV folds.",
    )

    args = parser.parse_args()

    run_story_9(
        input_path=args.input,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()