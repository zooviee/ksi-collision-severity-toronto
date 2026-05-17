"""
tests/test_ml_logistic_baseline.py
TDD test suite for Story 6 – Logistic Regression Baseline ML Classifier
Group 5 | DAMO-699-5

Cycle: RED → GREEN → BLUE per test
Run:   pytest tests/test_ml_logistic_baseline.py -v

TDD history:
  Test 1 (test_train_size_approximately_80_pct):
    RED   — module not importable (committed + pushed)
    GREEN — load_and_split implemented (committed + pushed)
    BLUE  — docstring added (committed + pushed)
  Tests 2-18:
    Written test-after — code already existed.
    All passing on first run.
"""

import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_logistic_baseline import (
    load_and_split,
    apply_smote,
    tune_regularisation,
    evaluate,
    CORE_FEATURES,
    TEST_SIZE,
    C_GRID,
    RANDOM_STATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

def _make_encoded_df(n: int = 500) -> pd.DataFrame:
    """
    Synthetic fixture with a learnable signal so the model beats random.
    older_adult=1 and hour < 6 increase fatal probability — model can learn this.
    """
    rng = np.random.default_rng(42)

    df = pd.DataFrame({f: rng.integers(0, 2, size=n).astype(float)
                       for f in CORE_FEATURES})
    df["invage"] = rng.uniform(18, 80, size=n)
    df["hour"]   = rng.integers(0, 24, size=n).astype(float)

    # Add learnable signal: older_adult and early hours predict fatality
    signal = (
        (df["older_adult"] == 1).astype(float) * 0.6 +
        (df["hour"] < 6).astype(float)          * 0.4
    )
    prob_fatal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-9)
    df["acclass_binary"] = (rng.random(n) < prob_fatal).astype(int)

    # Ensure both classes are present
    if df["acclass_binary"].sum() < 20:
        df.loc[:20, "acclass_binary"] = 1
    if (df["acclass_binary"] == 0).sum() < 20:
        df.loc[:20, "acclass_binary"] = 0

    return df


@pytest.fixture
def encoded_csv(tmp_path):
    df = _make_encoded_df(500)
    path = tmp_path / "ksi_encoded.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def split_data(encoded_csv):
    return load_and_split(encoded_csv)


@pytest.fixture
def smote_data(split_data):
    X_train, _, y_train, _, _, _, _ = split_data
    return apply_smote(X_train, y_train)


@pytest.fixture
def fitted_model(smote_data):
    X_smote, y_smote = smote_data
    return tune_regularisation(X_smote, y_smote)


# ─────────────────────────────────────────────────────────────────────────────
# load_and_split — 5 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAndSplit:

    def test_train_size_approximately_80_pct(self, encoded_csv):
        X_train, X_test, _, _, _, _, _ = load_and_split(encoded_csv)
        total     = len(X_train) + len(X_test)
        train_pct = len(X_train) / total
        assert abs(train_pct - 0.80) <= 0.02

    def test_no_overlap_between_train_and_test_indices(self, encoded_csv):
        _, _, _, _, train_idx, test_idx, _ = load_and_split(encoded_csv)
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0

    def test_stratification_preserves_fatal_rate(self, encoded_csv):
        _, _, y_train, y_test, _, _, _ = load_and_split(encoded_csv)
        train_rate = y_train.mean()
        test_rate  = y_test.mean()
        assert abs(train_rate - test_rate) < 0.01

    def test_raises_if_acclass_binary_missing(self, tmp_path):
        df = _make_encoded_df(200).drop(columns=["acclass_binary"])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="acclass_binary"):
            load_and_split(str(path))

    def test_reproducible_with_same_random_state(self, encoded_csv):
        r1 = load_and_split(encoded_csv)
        r2 = load_and_split(encoded_csv)
        assert r1[4] == r2[4]
        assert r1[5] == r2[5]


# ─────────────────────────────────────────────────────────────────────────────
# apply_smote — 4 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestApplySmote:

    def test_classes_balanced_after_smote(self, smote_data):
        _, y_res = smote_data
        counts = y_res.value_counts()
        assert counts[0] == counts[1]

    def test_training_size_increases(self, split_data, smote_data):
        X_train, _, y_train, _, _, _, _ = split_data
        X_res, _ = smote_data
        assert len(X_res) > len(X_train)

    def test_returns_dataframe_and_series(self, smote_data):
        X_res, y_res = smote_data
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)

    def test_no_nulls_in_smote_output(self, smote_data):
        X_res, y_res = smote_data
        assert X_res.isnull().sum().sum() == 0
        assert y_res.isnull().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# tune_regularisation — 3 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTuneRegularisation:

    def test_best_c_is_from_grid(self, fitted_model):
        _, best_C, _, _ = fitted_model
        assert best_C in C_GRID

    def test_best_auc_cv_in_valid_range(self, fitted_model):
        _, _, best_auc_cv, _ = fitted_model
        assert 0.5 <= best_auc_cv <= 1.0

    def test_model_has_predict_proba(self, fitted_model):
        model, _, _, _ = fitted_model
        assert hasattr(model, "predict_proba")


# ─────────────────────────────────────────────────────────────────────────────
# evaluate — 4 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluate:

    @pytest.fixture
    def metrics(self, fitted_model, split_data):
        model, _, _, _ = fitted_model
        _, X_test, _, y_test, _, _, _ = split_data
        return evaluate(model, X_test, y_test)

    def test_auc_above_random(self, metrics):
        assert metrics["auc"] > 0.5

    def test_confusion_matrix_shape(self, metrics):
        assert metrics["cm"].shape == (2, 2)

    def test_all_required_keys_present(self, metrics):
        for key in ["auc", "precision_macro", "recall_macro", "f1_macro",
                    "precision_fatal", "recall_fatal", "cm", "fpr", "tpr",
                    "y_prob", "y_pred"]:
            assert key in metrics

    def test_y_pred_is_binary(self, metrics):
        assert set(metrics["y_pred"]).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# Integration — 2 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_saves_expected_files(self, tmp_path, encoded_csv):
        from ml_logistic_baseline import run
        run(encoded_csv, str(tmp_path))
        for fname in ["train_indices.csv", "test_indices.csv",
                      "logistic_baseline_model.pkl",
                      "model_comparison_table.csv"]:
            assert (tmp_path / fname).exists(), f"'{fname}' not created"

    def test_no_index_overlap_in_saved_files(self, tmp_path, encoded_csv):
        from ml_logistic_baseline import run
        run(encoded_csv, str(tmp_path))
        train = set(pd.read_csv(tmp_path / "train_indices.csv")["train_index"])
        test  = set(pd.read_csv(tmp_path / "test_indices.csv")["test_index"])
        assert len(train & test) == 0