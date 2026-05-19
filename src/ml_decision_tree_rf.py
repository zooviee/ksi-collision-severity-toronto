"""
Story 7: Train and Evaluate Decision Tree and Random Forest Models

Tasks covered:
#39 - Train and tune Decision Tree classifier
#40 - Train and tune Random Forest classifier
#41 - Evaluate both models and save model comparison table
#42 - Extract and plot Random Forest feature importances
#43 - Plot Random Forest learning curves

Fix applied (2026-05-20):
  - REMOVED internal train_test_split() — Story 7 must inherit the exact
    same split produced by Story 6 (train_indices.csv / test_indices.csv).
  - REMOVED raw-CSV pipeline calls — Story 7 reads ksi_encoded.csv (already
    cleaned and OHE-encoded by Story 1), consistent with the README input
    file rules for Stories 7-9.
  - ADDED --indices-dir and --models-dir CLI arguments to match the run
    command documented in the README.
  - Filename is ml_decision_tree_rf.py so existing test imports
    continue to work without changes.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
)
from sklearn.tree import DecisionTreeClassifier

# Columns that must never be used as features — identifiers, target leakages,
# or high-cardinality free-text fields (mirrors Story 6's drop list).
DROP_COLUMNS = {
    "_id", "collision_id", "acclass", "accdate", "geometry",
    "stname1", "stname2", "stname3",
    "injury", "drivact", "drivcond", "road_user",
    "wardname", "neighbourhood", "division",
    "pedact", "pedcond", "pedtype",
    "cyclistype", "cycact", "cyccond",
    "manoeuvre", "safequip", "vehtype",
    "initdir", "visible", "fatal_no",
    "per_inv", "veh_no", "per_no",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_model_data(input_path, indices_dir):
    """
    Load ksi_encoded.csv and slice it using the train/test indices
    saved by Story 6.  This guarantees Stories 7, 8, and 9 all
    evaluate on exactly the same held-out set.

    Parameters
    ----------
    input_path : str | Path
        Path to outputs/story-1/ksi_encoded.csv
    indices_dir : str | Path
        Path to outputs/story-6/ (contains train_indices.csv and
        test_indices.csv)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    indices_dir = Path(indices_dir)

    # --- Load the encoded dataset (Story 1 output) ---
    df = pd.read_csv(input_path)

    # --- Load Story 6 split indices ---
    train_idx_path = indices_dir / "train_indices.csv"
    test_idx_path  = indices_dir / "test_indices.csv"

    if not train_idx_path.exists():
        raise FileNotFoundError(
            f"train_indices.csv not found in {indices_dir}. "
            "Run Story 6 (ml_logistic_baseline.py) first."
        )
    if not test_idx_path.exists():
        raise FileNotFoundError(
            f"test_indices.csv not found in {indices_dir}. "
            "Run Story 6 (ml_logistic_baseline.py) first."
        )

    train_idx = pd.read_csv(train_idx_path).squeeze("columns")
    test_idx  = pd.read_csv(test_idx_path).squeeze("columns")

    # --- Build feature matrix ---
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_cols + ["acclass_binary"], errors="ignore")
    X = X.select_dtypes(include=["number"]).fillna(0)
    y = df["acclass_binary"]

    # Slice using Story 6 indices (positional iloc — indices are row positions)
    X_train = X.iloc[train_idx.values]
    X_test  = X.iloc[test_idx.values]
    y_train = y.iloc[train_idx.values]
    y_test  = y.iloc[test_idx.values]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# TASK #39
# Train Decision Tree classifier.
# Tune max_depth (3–15) and min_samples_leaf (5–50)
# using GridSearchCV with 5-fold CV.
# Report best parameters.
# ---------------------------------------------------------
def train_decision_tree(X_train, y_train, cv=5):
    param_grid = {
        "max_depth":        [3, 5, 10, 15],
        "min_samples_leaf": [5, 10, 25, 50],
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


# ---------------------------------------------------------
# TASK #40
# Train Random Forest classifier.
# Tune n_estimators (100–500), max_depth (5–20),
# and min_samples_leaf using RandomizedSearchCV.
# Report best parameters and OOB score.
# ---------------------------------------------------------
def train_random_forest(X_train, y_train, cv=5, n_iter=10):
    param_dist = {
        "n_estimators":     [100, 200, 300, 400, 500],
        "max_depth":        [5, 10, 15, 20],
        "min_samples_leaf": [5, 10, 25, 50],
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            oob_score=True,
            bootstrap=True,
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
    )

    random_search.fit(X_train, y_train)

    best_model  = random_search.best_estimator_
    best_params = random_search.best_params_
    oob_score   = best_model.oob_score_

    return best_model, best_params, oob_score


# ---------------------------------------------------------
# TASK #41
# Evaluate both models on the test set using the same
# metrics table as the baseline.
# Add rows to the model comparison table.
# ---------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model":     model_name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }


def save_model_comparison(metrics_rows, output_dir, models_dir=None):
    """
    Save Story 7 metrics and, if models_dir is provided, append them to the
    shared model_comparison_table.csv that Story 6 started.
    """
    comparison_df = pd.DataFrame(metrics_rows)
    comparison_df.to_csv(
        output_dir / "task_41_model_comparison_tree_models.csv",
        index=False,
    )

    # Append to Story 6's shared comparison table when available
    if models_dir is not None:
        shared_path = Path(models_dir) / "model_comparison_table.csv"
        if shared_path.exists():
            existing = pd.read_csv(shared_path)
            combined = pd.concat([existing, comparison_df], ignore_index=True)
            combined.to_csv(shared_path, index=False)
        else:
            comparison_df.to_csv(shared_path, index=False)

    return comparison_df


# ---------------------------------------------------------
# TASK #42
# Extract and plot feature importances from Random Forest.
# Save top 15 features as a bar chart for dashboard/report.
# ---------------------------------------------------------
def plot_random_forest_feature_importance(model, feature_names, output_dir, top_n=15):
    importances = (
        pd.DataFrame({
            "feature":    feature_names,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    importances.to_csv(
        output_dir / "task_42_random_forest_top_15_feature_importances.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))
    plt.barh(importances["feature"][::-1], importances["importance"][::-1])
    plt.title("Random Forest Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "task_42_random_forest_feature_importance.png", dpi=300)
    plt.close()

    return importances


# ---------------------------------------------------------
# TASK #43
# Plot learning curves for Random Forest:
# training score vs validation score vs training set size.
# Used to diagnose overfitting or underfitting.
# ---------------------------------------------------------
def plot_random_forest_learning_curve(model, X_train, y_train, output_dir, cv=5):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    )

    train_mean      = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    learning_df = pd.DataFrame({
        "train_size":     train_sizes,
        "training_f1":    train_mean,
        "validation_f1":  validation_mean,
    })

    learning_df.to_csv(
        output_dir / "task_43_random_forest_learning_curve.csv",
        index=False,
    )

    plt.figure(figsize=(9, 5))
    plt.plot(train_sizes, train_mean,      marker="o", label="Training F1")
    plt.plot(train_sizes, validation_mean, marker="o", label="Validation F1")
    plt.title("Random Forest Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task_43_random_forest_learning_curve.png", dpi=300)
    plt.close()

    return learning_df


# ---------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------

def run_story_7(input_path, output_dir, indices_dir, models_dir=None):
    output_dir = ensure_output_dir(output_dir)

    # Task #39 / #40 — build features using Story 6 split
    X_train, X_test, y_train, y_test = build_model_data(input_path, indices_dir)

    # Task #39 — Decision Tree with GridSearchCV
    decision_tree_model, decision_tree_params = train_decision_tree(X_train, y_train)

    # Task #40 — Random Forest with RandomizedSearchCV
    random_forest_model, random_forest_params, oob_score = train_random_forest(
        X_train, y_train,
    )

    # Task #41 — Evaluate and save metrics
    metrics_rows = [
        evaluate_model(decision_tree_model, X_test, y_test, "Decision Tree"),
        evaluate_model(random_forest_model, X_test, y_test, "Random Forest"),
    ]

    comparison_df = save_model_comparison(metrics_rows, output_dir, models_dir)

    # Save best hyperparameters (tasks #39 and #40)
    pd.DataFrame([
        {
            "model":       "Decision Tree",
            "best_params": str(decision_tree_params),
            "oob_score":   None,
        },
        {
            "model":       "Random Forest",
            "best_params": str(random_forest_params),
            "oob_score":   oob_score,
        },
    ]).to_csv(output_dir / "task_39_40_best_parameters.csv", index=False)

    # Task #42 — Feature importance plot
    plot_random_forest_feature_importance(
        random_forest_model,
        X_train.columns,
        output_dir,
    )

    # Task #43 — Learning curve plot
    plot_random_forest_learning_curve(
        random_forest_model,
        X_train,
        y_train,
        output_dir,
    )

    print("Story 7 complete.")
    print("Decision Tree best parameters:", decision_tree_params)
    print("Random Forest best parameters:", random_forest_params)
    print(f"Random Forest OOB score:       {oob_score:.4f}")
    print()
    print(comparison_df.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Story 7 — Decision Tree and Random Forest classifiers"
    )

    parser.add_argument(
        "--input",
        default="outputs/story-1/ksi_encoded.csv",
        help="Path to ksi_encoded.csv produced by Story 1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/story-7",
        help="Folder to save Story 7 outputs",
    )
    parser.add_argument(
        "--indices-dir",
        required=True,
        help=(
            "Folder containing train_indices.csv and test_indices.csv "
            "produced by Story 6"
        ),
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help=(
            "Folder containing model_comparison_table.csv from Story 6 "
            "(optional — Story 7 metrics will be appended)"
        ),
    )

    args = parser.parse_args()

    run_story_7(
        input_path=args.input,
        output_dir=args.output_dir,
        indices_dir=args.indices_dir,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()