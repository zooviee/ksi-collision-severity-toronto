"""
tests/test_ml_logistic_baseline.py
TDD test suite for Story 6 – Logistic Regression Baseline ML Classifier
Group 5 | DAMO-699-5

Cycle: RED → GREEN → BLUE per test
Run:   pytest tests/test_ml_logistic_baseline.py -v
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_logistic_baseline import (
    load_and_split,
    CORE_FEATURES,
    TEST_SIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

def _make_encoded_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_fatal     = max(30, n // 5)
    n_non_fatal = n - n_fatal
    target = np.array([1] * n_fatal + [0] * n_non_fatal)
    rng.shuffle(target)
    df = pd.DataFrame({f: rng.integers(0, 2, size=n).astype(float)
                       for f in CORE_FEATURES})
    df["invage"]         = rng.uniform(18, 80, size=n)
    df["hour"]           = rng.integers(0, 24, size=n).astype(float)
    df["acclass_binary"] = target
    return df


@pytest.fixture
def encoded_csv(tmp_path):
    df = _make_encoded_df(500)
    path = tmp_path / "ksi_encoded.csv"
    df.to_csv(path, index=False)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# load_and_split — Test 1 of 5
# ─────────────────────────────────────────────────────────────────────────────
"""
class TestLoadAndSplit:

    def test_train_size_approximately_80_pct(self, encoded_csv):
        X_train, X_test, y_train, y_test, _, _, _ = load_and_split(encoded_csv)
        total     = len(X_train) + len(X_test)
        train_pct = len(X_train) / total
        assert abs(train_pct - 0.80) <= 0.02
"""