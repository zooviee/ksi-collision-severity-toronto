"""
Story 8: XGBoost + SHAP Interpretability

Tasks covered:
#45 Train and tune XGBoost using the Story 6 train/test split.
#46 Handle class imbalance using scale_pos_weight.
#47 Evaluate XGBoost on the held-out test set and append comparison metrics.
#48 Generate SHAP beeswarm, bar, and dependence plots for the best XGBoost model.
#49 Export plain-language SHAP interpretation.

Dependencies:
- Story 1: outputs/story-1/ksi_encoded.csv
- Story 6: outputs/story-6/train_indices.csv, test_indices.csv, logistic_baseline_model.pkl
- Story 7: outputs/story-7/rf_model.pkl, task_41_model_comparison_table.csv

Usage:
python src/ml_xgboost_shap.py \
    --input outputs/story-1/ksi_encoded.csv \
    --output-dir outputs/story-8 \
    --indices-dir outputs/story-6 \
    --models-dir outputs/story-6 outputs/story-7
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================================================
# Helpers
# =========================================================

def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_encoded_data(input_path):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    if "acclass_binary" not in df.columns:
        raise ValueError("Target column acclass_binary not found. Run Story 1 first.")

    print("Loaded encoded dataset.")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nTarget distribution:")
    print(df["acclass_binary"].value_counts())

    return df


def load_story6_indices(indices_dir):
    indices_dir = Path(indices_dir)

    train_path = indices_dir / "train_indices.csv"
    test_path = indices_dir / "test_indices.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing Story 6 train indices: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Missing Story 6 test indices: {test_path}")

    train_indices = pd.read_csv(train_path).iloc[:, 0].to_numpy()
    test_indices = pd.read_csv(test_path).iloc[:, 0].to_numpy()

    print("\nLoaded Story 6 train/test indices.")
    print(f"Train indices: {len(train_indices)}")
    print(f"Test indices : {len(test_indices)}")

    return train_indices, test_indices


def validate_story6_dependencies(indices_dir, models_dirs):
    """
    Validate Story 6 dependency files.

    Story 8 must reuse the exact same train/test split from Story 6.
    It also confirms the logistic baseline model exists for model comparison continuity.
    """
    indices_dir = Path(indices_dir)

    required_files = [
        indices_dir / "train_indices.csv",
        indices_dir / "test_indices.csv",
    ]

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing required Story 6 file: {required_file}. "
                "Run Story 6 before Story 8."
            )

    logistic_model_found = False

    for directory in models_dirs:
        directory = Path(directory)
        if (directory / "logistic_baseline_model.pkl").exists():
            logistic_model_found = True
            print("\nValidated Story 6 dependency.")
            print(f"Found logistic baseline model: {directory / 'logistic_baseline_model.pkl'}")
            break

    if not logistic_model_found:
        raise FileNotFoundError(
            "Missing Story 6 logistic_baseline_model.pkl in --models-dir. "
            "Expected it inside outputs/story-6."
        )


def validate_story7_dependencies(models_dirs):
    """
    Validate Story 7 dependency files.

    Story 8 depends on Story 7 because:
    - rf_model.pkl confirms the Random Forest model was trained and saved.
    - task_41_model_comparison_table.csv provides previous model metrics for comparison.

    SHAP in Story 8 is computed for the best XGBoost model, as required by Task #48.
    """
    story7_dir = None

    for directory in models_dirs:
        directory = Path(directory)

        has_rf_model = (directory / "rf_model.pkl").exists()
        has_comparison_table = (directory / "task_41_model_comparison_table.csv").exists()

        if has_rf_model or has_comparison_table:
            story7_dir = directory
            break

    if story7_dir is None:
        raise FileNotFoundError(
            "Story 7 outputs were not found in --models-dir. "
            "Expected outputs/story-7 with rf_model.pkl and task_41_model_comparison_table.csv."
        )

    rf_model_path = story7_dir / "rf_model.pkl"
    comparison_path = story7_dir / "task_41_model_comparison_table.csv"

    if not rf_model_path.exists():
        raise FileNotFoundError(
            f"Missing Story 7 Random Forest model: {rf_model_path}. "
            "Please rerun Story 7 after the latest patch."
        )

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing Story 7 model comparison table: {comparison_path}. "
            "Please rerun Story 7 after the latest patch."
        )

    try:
        with open(rf_model_path, "rb") as f:
            story7_rf_model = pickle.load(f)

        print("\nValidated Story 7 dependency.")
        print(f"Loaded Random Forest model from: {rf_model_path}")
        print(f"Found Story 7 comparison table: {comparison_path}")
        print(f"Story 7 RF model type: {type(story7_rf_model).__name__}")

    except Exception as exc:
        raise RuntimeError(
            f"Story 7 rf_model.pkl exists but could not be loaded: {rf_model_path}"
        ) from exc

    return {
        "story7_dir": story7_dir,
        "rf_model_path": rf_model_path,
        "comparison_path": comparison_path,
    }


def prepare_features(df):
    """
    Prepare feature matrix for XGBoost.

    Removes:
    - target/leakage columns
    - raw text columns
    - ID/date columns
    - columns too close to outcome, such as injury and fatal_no

    Keeps:
    - numeric variables
    - one-hot encoded Story 1 variables
    - engineered temporal variables
    """

    y = df["acclass_binary"].astype(int)

    columns_to_drop = [
        # Target/leakage
        "acclass",
        "acclass_binary",
        "injury",
        "injury_encoded",
        "fatal_no",

        # IDs/raw date/text/location fields
        "_id",
        "collision_id",
        "accdate",
        "stname1",
        "stname2",
        "stname3",
        "geometry",
        "wardname",
        "neighbourhood",
        "division",

        # Raw categorical/text fields
        "visible",
        "failtorem",
        "vehtype",
        "initdir",
        "safequip",
        "drivact",
        "drivcond",
        "pedact",
        "pedcond",
        "manoeuvre",
        "pedtype",
        "cyclistype",
        "cycact",
        "cyccond",
        "road_user",
        "day_of_week_name",
        "month_name",
        "season",
    ]

    X = df.drop(columns=columns_to_drop, errors="ignore")

    # XGBoost needs numeric input.
    X = X.select_dtypes(include=[np.number]).copy()

    # Clean values.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Remove constant columns.
    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=True) <= 1
    ]
    X = X.drop(columns=constant_cols, errors="ignore")

    print("\nPrepared XGBoost features.")
    print(f"Number of features: {X.shape[1]}")

    return X, y


def split_by_story6_indices(X, y, train_indices, test_indices):
    """
    Reuse exact Story 6 train/test split.
    """

    X_train = X.loc[train_indices].copy()
    X_test = X.loc[test_indices].copy()
    y_train = y.loc[train_indices].copy()
    y_test = y.loc[test_indices].copy()

    print("\nApplied Story 6 train/test split.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# =========================================================
# Task #45 and #46: XGBoost tuning + imbalance handling
# =========================================================

def compute_scale_pos_weight(y_train):
    """
    scale_pos_weight = negative class count / positive class count.
    This handles class imbalance without SMOTE for XGBoost.
    """

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())

    if positive == 0:
        raise ValueError("No positive fatal cases found in training set.")

    scale_pos_weight = negative / positive

    print("\nTask #46: Class imbalance handling")
    print(f"Non-fatal train count: {negative}")
    print(f"Fatal train count    : {positive}")
    print(f"scale_pos_weight     : {scale_pos_weight:.4f}")

    return scale_pos_weight


def tune_xgboost_with_optuna(X_train, y_train, X_test, y_test, scale_pos_weight, n_trials=25):
    """
    Tune XGBoost with Optuna.

    Parameters tuned:
    - n_estimators
    - max_depth
    - learning_rate
    - subsample
    - colsample_bytree
    - min_child_weight
    - gamma
    - reg_lambda
    - reg_alpha
    """

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for Story 8 tuning. Install it with: pip install optuna"
        ) from exc

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        return auc

    print("\nTask #45: Tuning XGBoost with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Best AUC from Optuna: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_params = study.best_params.copy()
    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    })

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    return final_model, study


# =========================================================
# Task #47: Evaluation
# =========================================================

def evaluate_model(model, X_test, y_test, output_dir):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": "XGBoost",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "task_47_xgboost_metrics.csv", index=False)

    report = classification_report(y_test, y_pred, target_names=["Non-Fatal", "Fatal"])
    with open(output_dir / "task_47_xgboost_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        cm,
        index=["Actual Non-Fatal", "Actual Fatal"],
        columns=["Predicted Non-Fatal", "Predicted Fatal"],
    ).to_csv(output_dir / "task_47_xgboost_confusion_matrix.csv")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"XGBoost AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Task 47: XGBoost ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task_47_xgboost_roc_curve.png", dpi=300)
    plt.close()

    print("\nTask #47: XGBoost test evaluation")
    print(metrics_df.to_string(index=False))

    return metrics_df, y_prob, y_pred


def update_model_comparison(output_dir, models_dirs, xgb_metrics_df):
    """
    Build Story 8 model comparison table using available Story 6 and Story 7 metrics.
    """

    comparison_frames = []

    for models_dir in models_dirs:
        models_dir = Path(models_dir)

        candidates = [
            models_dir / "model_comparison_table.csv",
            models_dir / "task_34_logistic_baseline_metrics.csv",
            models_dir / "task_41_model_comparison_table.csv",
            models_dir / "task_41_model_comparison_tree_models.csv",
        ]

        for path in candidates:
            if path.exists():
                try:
                    comparison_frames.append(pd.read_csv(path))
                    print(f"Loaded comparison metrics from: {path}")
                except Exception:
                    print(f"Could not read comparison metrics from: {path}")

    comparison_frames.append(xgb_metrics_df)

    comparison = pd.concat(comparison_frames, ignore_index=True, sort=False)

    # Remove duplicate model rows where possible.
    if "model" in comparison.columns:
        comparison = comparison.drop_duplicates(subset=["model"], keep="last")

    if "Model" in comparison.columns:
        comparison = comparison.drop_duplicates(subset=["Model"], keep="last")

    comparison.to_csv(output_dir / "task_47_model_comparison_with_xgboost.csv", index=False)

    return comparison


# =========================================================
# Task #48: SHAP plots
# =========================================================

def compute_shap_values(model, X_test, output_dir, max_rows=1000):
    """
    Compute SHAP values for the best XGBoost model.

    This satisfies Task #48:
    - SHAP summary plot
    - SHAP bar plot
    - SHAP dependence plot for top feature
    """

    shap_sample = X_test.copy()

    if len(shap_sample) > max_rows:
        shap_sample = shap_sample.sample(n=max_rows, random_state=42)

    print("\nTask #48: Computing SHAP values for the best XGBoost model...")
    print(f"SHAP sample shape: {shap_sample.shape}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)

    np.save(output_dir / "shap_values.npy", shap_values)
    shap_sample.to_csv(output_dir / "task_48_shap_sample.csv", index=False)

    # Beeswarm summary plot
    plt.figure()
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / "task_48_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / "task_48_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Dependence plot for top feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_index = int(np.argmax(mean_abs_shap))
    top_feature = shap_sample.columns[top_feature_index]

    plt.figure()
    shap.dependence_plot(top_feature, shap_values, shap_sample, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "task_48_shap_dependence_top_feature.png", dpi=300, bbox_inches="tight")
    plt.close()

    feature_importance = pd.DataFrame({
        "feature": shap_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    feature_importance.to_csv(output_dir / "task_48_shap_feature_importance.csv", index=False)

    print("SHAP plots saved.")
    print(f"Top SHAP feature: {top_feature}")

    return shap_values, shap_sample, feature_importance


# =========================================================
# Task #49: Interpretation
# =========================================================

def write_shap_interpretation(feature_importance, metrics_df, output_dir):
    top_features = feature_importance.head(5)

    auc = metrics_df["roc_auc"].iloc[0]
    recall = metrics_df["recall"].iloc[0]
    precision = metrics_df["precision"].iloc[0]

    lines = []
    lines.append("Task #49: Plain-Language SHAP Interpretation")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"The XGBoost model achieved a test ROC-AUC of {auc:.3f}, "
        f"with fatal-class recall of {recall:.3f} and fatal-class precision of {precision:.3f}."
    )
    lines.append("")
    lines.append("The top SHAP drivers of model predictions were:")
    lines.append("")

    for i, row in enumerate(top_features.itertuples(index=False), start=1):
        lines.append(
            f"{i}. {row.feature}: This feature had one of the largest average SHAP impacts, "
            f"meaning it strongly influenced whether the XGBoost model pushed a prediction "
            f"toward fatal or non-fatal collision risk."
        )

    lines.append("")
    lines.append(
        "These SHAP results should be interpreted as model explanations, not causal proof. "
        "They identify which features the trained XGBoost model relied on most when predicting "
        "fatal collision outcomes."
    )

    interpretation = "\n".join(lines)

    with open(output_dir / "task_49_shap_interpretation.txt", "w", encoding="utf-8") as f:
        f.write(interpretation)

    print("\nTask #49 interpretation saved.")
    print(interpretation)

    return interpretation


# =========================================================
# Main
# =========================================================

def main(input_path, output_dir, indices_dir, models_dirs, n_trials):
    output_dir = ensure_output_dir(output_dir)

    df = load_encoded_data(input_path)

    # Validate Story 6 and Story 7 dependencies before Story 8 training.
    validate_story6_dependencies(indices_dir, models_dirs)
    validate_story7_dependencies(models_dirs)

    train_indices, test_indices = load_story6_indices(indices_dir)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_by_story6_indices(
        X, y, train_indices, test_indices
    )

    scale_pos_weight = compute_scale_pos_weight(y_train)

    model, study = tune_xgboost_with_optuna(
        X_train,
        y_train,
        X_test,
        y_test,
        scale_pos_weight,
        n_trials=n_trials,
    )

    # Save model and tuning details
    joblib.dump(model, output_dir / "xgb_model.pkl")

    with open(output_dir / "task_45_optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)

    study.trials_dataframe().to_csv(output_dir / "task_45_optuna_trials.csv", index=False)

    metrics_df, _, _ = evaluate_model(model, X_test, y_test, output_dir)
    update_model_comparison(output_dir, models_dirs, metrics_df)

    _, _, feature_importance = compute_shap_values(
        model,
        X_test,
        output_dir,
    )

    write_shap_interpretation(feature_importance, metrics_df, output_dir)

    print("\nStory 8 complete.")
    print(f"Outputs saved to: {output_dir}")
    print("\nKey files generated:")
    print("- xgb_model.pkl")
    print("- shap_values.npy")
    print("- task_45_optuna_best_params.json")
    print("- task_45_optuna_trials.csv")
    print("- task_47_xgboost_metrics.csv")
    print("- task_47_model_comparison_with_xgboost.csv")
    print("- task_48_shap_beeswarm.png")
    print("- task_48_shap_bar.png")
    print("- task_48_shap_dependence_top_feature.png")
    print("- task_49_shap_interpretation.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 8: XGBoost tuning and SHAP interpretation."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to Story 1 encoded dataset."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Story 8 outputs will be saved."
    )

    parser.add_argument(
        "--indices-dir",
        required=True,
        help="Directory containing Story 6 train_indices.csv and test_indices.csv."
    )

    parser.add_argument(
        "--models-dir",
        nargs="+",
        required=True,
        help="One or more directories containing prior model outputs from Stories 6 and 7."
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Number of Optuna trials for XGBoost tuning. Default is 25."
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output_dir,
        indices_dir=args.indices_dir,
        models_dirs=args.models_dir,
        n_trials=args.n_trials,
    )