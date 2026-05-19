from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_decision_tree(X_train, y_train, cv=5):
    param_grid = {
        "max_depth": [3, 5, 10, 15],
        "min_samples_leaf": [5, 10, 25, 50],
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_


def train_random_forest(X_train, y_train, cv=5, n_iter=10):
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [5, 10, 15, 20],
        "min_samples_leaf": [5, 10, 25, 50],
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(
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

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_estimator_.oob_score_


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }