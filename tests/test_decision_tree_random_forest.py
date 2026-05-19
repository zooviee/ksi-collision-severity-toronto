import pandas as pd

from decision_tree_random_forest import (
    train_decision_tree,
    train_random_forest,
    evaluate_model,
)


def sample_ml_data():
    X_train = pd.DataFrame({
        "hour": [8, 9, 10, 11, 12, 13, 14, 15],
        "invage": [20, 25, 30, 35, 40, 45, 50, 55],
        "light_Dark": [0, 0, 1, 1, 0, 1, 0, 1],
    })
    y_train = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])

    X_test = pd.DataFrame({
        "hour": [16, 17, 18, 19],
        "invage": [60, 65, 70, 75],
        "light_Dark": [0, 1, 0, 1],
    })
    y_test = pd.Series([0, 1, 0, 1])

    return X_train, X_test, y_train, y_test


def test_decision_tree_training_returns_model_and_best_params():
    X_train, _, y_train, _ = sample_ml_data()

    model, best_params = train_decision_tree(X_train, y_train, cv=2)

    assert model is not None
    assert "max_depth" in best_params
    assert "min_samples_leaf" in best_params


def test_random_forest_training_returns_model_best_params_and_oob_score():
    X_train, _, y_train, _ = sample_ml_data()

    model, best_params, oob_score = train_random_forest(
        X_train,
        y_train,
        cv=2,
        n_iter=2,
    )

    assert model is not None
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert "min_samples_leaf" in best_params
    assert oob_score is not None


def test_evaluate_model_returns_metrics_dictionary():
    X_train, X_test, y_train, y_test = sample_ml_data()

    model, _ = train_decision_tree(X_train, y_train, cv=2)
    metrics = evaluate_model(model, X_test, y_test, "Decision Tree")

    assert metrics["model"] == "Decision Tree"
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics