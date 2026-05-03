"""
tests/test_data_preparation.py
TDD test suite for Story 1 – KSI Data Preparation Pipeline
Group 5 | DAMO-699-5

Run with:
    pytest tests/test_data_preparation.py -v

All tests use a small synthetic fixture DataFrame that mirrors the
real KSI dataset schema — no internet or file-system dependency.
"""

import numpy as np
import pandas as pd
import pytest

# ── Import all pipeline functions ─────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preparation import (
    KEY_VARS,
    VARIABLE_CATALOGUE,
    OHE_VARS,
    MODE_IMPUTE_VARS,
    BEHAVIOURAL_FLAG_VARS,
    INJURY_ORDER,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    _month_to_season,
    document_variables,
    audit_missing,
    impute_and_flag,
    encode_categoricals,
    engineer_temporal,
    encode_target,
    split_and_oversample,
    validate_dataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 200) -> pd.DataFrame:
    """
    Build a synthetic DataFrame that mirrors the real KSI schema.
    n must be ≥ 50 so SMOTE has enough minority samples.
    """
    rng = np.random.default_rng(42)

    # Ensure meaningful minority class (fatal ≈ 20%)
    n_fatal     = max(10, n // 5)
    n_non_fatal = n - n_fatal
    acclass = (["Fatal Injury"] * n_fatal +
               ["Non-Fatal Injury"] * n_non_fatal)
    rng.shuffle(acclass)

    dates = pd.date_range("2010-01-01", periods=n, freq="7D")

    lights     = ["Daylight", "Dark", "Dark with Artificial Lighting",
                  "Dusk", "Dawn", None]
    rdsfconds  = ["Dry", "Wet", "Ice", "Loose Snow", None]
    traffictls = ["No Control", "Traffic Signal", "Stop Sign", None]
    road_class = ["Major Arterial", "Minor Arterial", "Collector", None]
    acclocos   = ["At Intersection", "Non-Intersection", None]
    impacts    = ["Angle", "Rear End", "Turning Movement", None]
    injuries   = ["Fatal", "Major", "Minor", "Minimal", None]
    drivacts   = ["Driving properly", "Improper turn", None]
    driveconds = ["Normal", "Inattentive", "Had Been Drinking", None]
    road_users = ["driver", "passenger", "pedestrian", "cyclist"]
    wards      = [f"Ward {i}" for i in range(1, 6)]
    hoods      = [f"Neighbourhood {i}" for i in range(1, 6)]

    def _choice(pool, size):
        return [pool[i % len(pool)] for i in range(size)]

    df = pd.DataFrame({
        "_id":            range(1, n + 1),
        "collision_id":   [f"2010:{i}" for i in range(n)],
        "accdate":        [d.isoformat() for d in dates],
        "stname1":        ["YONGE ST"] * n,
        "stname2":        ["BLOOR ST"] * n,
        "stname3":        [None] * n,
        "per_inv":        rng.integers(1, 6, size=n).tolist(),
        "acclass":        acclass,
        "accloc":         _choice(acclocos, n),
        "traffictl":      _choice(traffictls, n),
        "impactype":      _choice(impacts, n),
        "visible":        ["Clear"] * n,
        "light":          _choice(lights, n),
        "rdsfcond":       _choice(rdsfconds, n),
        "road_class":     _choice(road_class, n),
        "failtorem":      ["false"] * n,
        "longitude":      rng.uniform(LON_MIN, LON_MAX, size=n).tolist(),
        "latitude":       rng.uniform(LAT_MIN, LAT_MAX, size=n).tolist(),
        "veh_no":         rng.integers(1, 3, size=n).tolist(),
        "vehtype":        ["Automobile or Station Wagon"] * n,
        "initdir":        ["North"] * n,
        "per_no":         rng.integers(1, 4, size=n).tolist(),
        "invage":         rng.choice([20.0, 35.0, 55.0, 70.0, np.nan], size=n).tolist(),
        "injury":         _choice(injuries, n),
        "safequip":       ["Lap and Shoulder Belt"] * n,
        "drivact":        _choice(drivacts, n),
        "drivcond":       _choice(driveconds, n),
        "pedact":         [None] * n,
        "pedcond":        [None] * n,
        "manoeuvre":      ["Going Ahead"] * n,
        "pedtype":        [None] * n,
        "cyclistype":     [None] * n,
        "cycact":         [None] * n,
        "cyccond":        [None] * n,
        "road_user":      _choice(road_users, n),
        "fatal_no":       rng.integers(0, 2, size=n).tolist(),
        "wardname":       _choice(wards, n),
        "division":       ["55 Division"] * n,
        "neighbourhood":  _choice(hoods, n),
        "aggressive":     rng.choice([True, False], size=n).tolist(),
        "distracted":     rng.choice([True, False], size=n).tolist(),
        "cyclist":        rng.choice([True, False], size=n).tolist(),
        "motorcyclist":   rng.choice([True, False], size=n).tolist(),
        "other_micromobility": [False] * n,
        "older_adult":    rng.choice([True, False], size=n).tolist(),
        "pedestrian":     rng.choice([True, False], size=n).tolist(),
        "red_light":      rng.choice([True, False], size=n).tolist(),
        "school_child":   [False] * n,
        "heavy_truck":    [False] * n,
        "geometry":       [None] * n,
    })
    return df


@pytest.fixture
def raw_df():
    return _make_raw_df(200)


@pytest.fixture
def df_imputed(raw_df):
    return impute_and_flag(raw_df)


@pytest.fixture
def df_temporal(df_imputed):
    return engineer_temporal(df_imputed)


@pytest.fixture
def df_target(df_temporal):
    return encode_target(df_temporal)


@pytest.fixture
def df_encoded(df_target):
    return encode_categoricals(df_target)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadDataset:
    def test_key_vars_present_in_raw(self, raw_df):
        """All 24 key variables from the proposal must be in the raw dataset."""
        for var in KEY_VARS:
            assert var in raw_df.columns, f"Key variable '{var}' missing from dataset."

    def test_raw_df_non_empty(self, raw_df):
        assert len(raw_df) > 0

    def test_raw_df_has_expected_columns(self, raw_df):
        assert raw_df.shape[1] >= 40  # KSI dataset has ~50 columns


# ─────────────────────────────────────────────────────────────────────────────
# 2. Variable catalogue
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableCatalogue:
    def test_catalogue_covers_all_key_vars(self):
        for var in KEY_VARS:
            assert var in VARIABLE_CATALOGUE, f"'{var}' missing from VARIABLE_CATALOGUE."

    def test_catalogue_has_required_fields(self):
        required_fields = {"type", "role", "nullable", "values"}
        for var, meta in VARIABLE_CATALOGUE.items():
            missing = required_fields - set(meta.keys())
            assert not missing, f"'{var}' catalogue entry missing fields: {missing}"

    def test_document_variables_returns_dataframe(self, raw_df):
        cat = document_variables(raw_df)
        assert isinstance(cat, pd.DataFrame)
        assert len(cat) == len(VARIABLE_CATALOGUE)

    def test_document_variables_has_pct_null(self, raw_df):
        cat = document_variables(raw_df)
        assert "pct_null" in cat.columns

    def test_boolean_vars_marked_not_nullable(self):
        bool_vars = ["aggressive", "distracted", "pedestrian", "cyclist",
                     "motorcyclist", "red_light", "older_adult", "school_child"]
        for var in bool_vars:
            assert VARIABLE_CATALOGUE[var]["nullable"] is False, \
                f"'{var}' should be marked nullable=False."


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missingness audit
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingnessAudit:
    def test_audit_returns_dataframe(self, raw_df):
        report = audit_missing(raw_df)
        assert isinstance(report, pd.DataFrame)

    def test_audit_has_required_columns(self, raw_df):
        report = audit_missing(raw_df)
        for col in ("column", "n_missing", "pct_missing", "flag_high_missing"):
            assert col in report.columns

    def test_audit_covers_all_columns(self, raw_df):
        report = audit_missing(raw_df)
        assert len(report) == raw_df.shape[1]

    def test_pct_missing_bounds(self, raw_df):
        report = audit_missing(raw_df)
        assert report["pct_missing"].between(0, 100).all()

    def test_flag_column_is_boolean(self, raw_df):
        report = audit_missing(raw_df)
        assert report["flag_high_missing"].dtype == bool

    def test_high_missing_flag_triggers_at_20_pct(self):
        """Create a column that is 25% null — it must be flagged."""
        df = pd.DataFrame({"a": [None] * 25 + [1] * 75})
        report = audit_missing(df)
        flagged_row = report[report["column"] == "a"]
        assert bool(flagged_row["flag_high_missing"].iloc[0]) is True

    def test_low_missing_not_flagged(self):
        """A column with 5% nulls must NOT be flagged."""
        df = pd.DataFrame({"a": [None] * 5 + [1] * 95})
        report = audit_missing(df)
        flagged_row = report[report["column"] == "a"]
        assert bool(flagged_row["flag_high_missing"].iloc[0]) is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Imputation and behavioural flagging
# ─────────────────────────────────────────────────────────────────────────────

class TestImputation:
    def test_mode_impute_removes_nulls(self, raw_df):
        df_out = impute_and_flag(raw_df)
        for col in MODE_IMPUTE_VARS:
            if col in raw_df.columns and raw_df[col].isna().sum() > 0:
                assert df_out[col].isna().sum() == 0, \
                    f"Nulls remain in '{col}' after mode imputation."

    def test_invage_nulls_imputed(self, raw_df):
        df_out = impute_and_flag(raw_df)
        assert df_out["invage"].isna().sum() == 0

    def test_behavioural_flag_columns_created(self, raw_df):
        df_out = impute_and_flag(raw_df)
        for col in BEHAVIOURAL_FLAG_VARS:
            if col in raw_df.columns:
                assert f"{col}_missing" in df_out.columns, \
                    f"Missing indicator '{col}_missing' not created."

    def test_behavioural_flag_is_binary(self, raw_df):
        df_out = impute_and_flag(raw_df)
        for col in BEHAVIOURAL_FLAG_VARS:
            indicator = f"{col}_missing"
            if indicator in df_out.columns:
                bad = set(df_out[indicator].unique()) - {0, 1}
                assert not bad, f"'{indicator}' contains non-binary values: {bad}"

    def test_raw_df_unchanged(self, raw_df):
        """impute_and_flag must not mutate the input DataFrame."""
        raw_copy = raw_df.copy(deep=True)
        _ = impute_and_flag(raw_df)
        pd.testing.assert_frame_equal(raw_df.reset_index(drop=True),
                                      raw_copy.reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Categorical encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestCategoricalEncoding:
    def test_ohe_columns_created(self, df_encoded):
        """OHE should expand each nominal column into multiple indicator columns."""
        for col in OHE_VARS:
            ohe_cols = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
            assert len(ohe_cols) >= 1, \
                f"No OHE columns found for '{col}'."

    def test_original_ohe_cols_removed(self, df_encoded):
        for col in OHE_VARS:
            assert col not in df_encoded.columns, \
                f"Original column '{col}' still present after OHE."

    def test_injury_encoded_column_exists(self, df_encoded):
        assert "injury_encoded" in df_encoded.columns

    def test_injury_encoded_values_in_range(self, df_encoded):
        valid_values = set(range(len(INJURY_ORDER))) | {np.nan}
        actual = set(df_encoded["injury_encoded"].dropna().unique())
        assert actual.issubset(set(range(len(INJURY_ORDER)))), \
            f"injury_encoded has unexpected values: {actual}"

    def test_boolean_cols_are_int(self, df_encoded):
        bool_flag_cols = ["aggressive", "distracted", "pedestrian", "cyclist",
                          "motorcyclist", "red_light", "older_adult", "school_child"]
        for col in bool_flag_cols:
            if col in df_encoded.columns:
                assert df_encoded[col].dtype in [np.int64, np.int32, int], \
                    f"'{col}' should be int after bool conversion."

    def test_ohe_columns_are_numeric(self, df_encoded):
        ohe_generated = [c for base in OHE_VARS
                         for c in df_encoded.columns if c.startswith(f"{base}_")]
        for col in ohe_generated:
            assert pd.api.types.is_numeric_dtype(df_encoded[col]), \
                f"OHE column '{col}' is not numeric."


# ─────────────────────────────────────────────────────────────────────────────
# 6. Temporal feature engineering
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_all_temporal_columns_created(self, df_temporal):
        expected = ["hour", "day_of_week", "day_of_week_name",
                    "month", "month_name", "year", "season", "is_weekend"]
        for col in expected:
            assert col in df_temporal.columns, f"Temporal column '{col}' missing."

    def test_hour_range(self, df_temporal):
        assert df_temporal["hour"].between(0, 23).all()

    def test_day_of_week_range(self, df_temporal):
        assert df_temporal["day_of_week"].between(0, 6).all()

    def test_month_range(self, df_temporal):
        assert df_temporal["month"].between(1, 12).all()

    def test_year_plausible(self, df_temporal):
        assert df_temporal["year"].between(2006, 2026).all()

    def test_season_values(self, df_temporal):
        valid_seasons = {"Winter", "Spring", "Summer", "Fall"}
        assert set(df_temporal["season"].dropna().unique()).issubset(valid_seasons)

    def test_is_weekend_binary(self, df_temporal):
        bad = set(df_temporal["is_weekend"].unique()) - {0, 1}
        assert not bad

    def test_season_mapping_winter(self):
        months = pd.Series([12, 1, 2])
        result = _month_to_season(months)
        assert all(result == "Winter")

    def test_season_mapping_spring(self):
        months = pd.Series([3, 4, 5])
        result = _month_to_season(months)
        assert all(result == "Spring")

    def test_season_mapping_summer(self):
        months = pd.Series([6, 7, 8])
        result = _month_to_season(months)
        assert all(result == "Summer")

    def test_season_mapping_fall(self):
        months = pd.Series([9, 10, 11])
        result = _month_to_season(months)
        assert all(result == "Fall")

    def test_is_weekend_correct_days(self, df_temporal):
        """day_of_week 5 (Sat) and 6 (Sun) must map to is_weekend=1."""
        weekends = df_temporal[df_temporal["day_of_week"].isin([5, 6])]
        assert (weekends["is_weekend"] == 1).all()

    def test_is_weekday_correct(self, df_temporal):
        weekdays = df_temporal[df_temporal["day_of_week"] < 5]
        assert (weekdays["is_weekend"] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Target encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetEncoding:
    def test_acclass_binary_exists(self, df_target):
        assert "acclass_binary" in df_target.columns

    def test_fatal_encoded_as_1(self, df_target):
        fatal_rows = df_target[df_target["acclass"] == "Fatal Injury"]
        assert (fatal_rows["acclass_binary"] == 1).all()

    def test_non_fatal_encoded_as_0(self, df_target):
        non_fatal_rows = df_target[df_target["acclass"] == "Non-Fatal Injury"]
        assert (non_fatal_rows["acclass_binary"] == 0).all()

    def test_property_damage_dropped(self, raw_df):
        """Property Damage Only rows must be removed before encoding."""
        raw_with_pd = raw_df.copy()
        raw_with_pd.loc[0, "acclass"] = "Property Damage Only"
        df_temporal = engineer_temporal(impute_and_flag(raw_with_pd))
        df_t = encode_target(df_temporal)
        assert "Property Damage Only" not in df_t["acclass"].values

    def test_target_is_binary(self, df_target):
        bad = set(df_target["acclass_binary"].unique()) - {0, 1}
        assert not bad

    def test_both_classes_present(self, df_target):
        assert 0 in df_target["acclass_binary"].values
        assert 1 in df_target["acclass_binary"].values

    def test_imbalance_ratio_computable(self, df_target):
        counts = df_target["acclass_binary"].value_counts()
        ratio = counts[0] / counts[1]
        assert ratio > 1  # majority class > minority class


# ─────────────────────────────────────────────────────────────────────────────
# 8. SMOTE oversampling
# ─────────────────────────────────────────────────────────────────────────────

class TestSMOTE:
    @pytest.fixture
    def splits(self, df_encoded):
        feature_cols = [
            c for c in df_encoded.columns
            if c not in {"_id", "collision_id", "stname1", "stname2", "stname3",
                         "geometry", "acclass", "acclass_binary", "accdate",
                         "day_of_week_name", "month_name", "season",
                         "injury", "drivact", "drivcond", "road_user",
                         "wardname", "neighbourhood", "division"}
            and pd.api.types.is_numeric_dtype(df_encoded[c])
        ]
        return split_and_oversample(df_encoded, feature_cols)

    def test_split_keys_present(self, splits):
        for key in ("X_train", "X_test", "y_train", "y_test",
                    "X_train_smote", "y_train_smote"):
            assert key in splits, f"Key '{key}' missing from split output."

    def test_smote_increases_training_size(self, splits):
        assert len(splits["X_train_smote"]) >= len(splits["X_train"])

    def test_smote_balances_classes(self, splits):
        counts = splits["y_train_smote"].value_counts()
        # After SMOTE minority class count should equal majority
        assert counts[0] == counts[1]

    def test_test_set_unchanged_by_smote(self, splits):
        """Test set must not be augmented — size should equal original split size."""
        original_test_size = len(splits["X_test"])
        assert original_test_size > 0
        # Test set size should be ~20% of the pre-SMOTE data (not affected by SMOTE)
        total_original = len(splits["X_train"]) + len(splits["X_test"])
        expected_test_size = int(total_original * 0.2)
        # Allow ±5 rows for rounding
        assert abs(original_test_size - expected_test_size) <= 5

    def test_no_smote_samples_in_test(self, splits):
        """
        SMOTE samples should only appear in X_train_smote.
        The test set must have fewer rows than the SMOTE training set.
        """
        assert len(splits["X_test"]) < len(splits["X_train_smote"])

    def test_smote_returns_dataframe(self, splits):
        assert isinstance(splits["X_train_smote"], pd.DataFrame)
        assert isinstance(splits["y_train_smote"], pd.Series)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Final validation assertions
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:
    @pytest.fixture
    def valid_df_and_features(self, df_encoded):
        drop_cols = {
            "_id", "collision_id", "stname1", "stname2", "stname3",
            "geometry", "acclass", "acclass_binary", "accdate",
            "day_of_week_name", "month_name", "season",
            "injury", "drivact", "drivcond", "road_user",
            "wardname", "neighbourhood", "division",
        }
        feature_cols = [
            c for c in df_encoded.columns
            if c not in drop_cols
            and pd.api.types.is_numeric_dtype(df_encoded[c])
        ]
        return df_encoded, feature_cols

    def test_validation_passes_on_clean_data(self, valid_df_and_features):
        df, feature_cols = valid_df_and_features
        # Should not raise
        validate_dataset(df, feature_cols)

    def test_validation_fails_on_null_imputed_col(self, valid_df_and_features):
        df, feature_cols = valid_df_and_features
        # Inject nulls into a column that should be clean after imputation
        df_bad = df.copy()
        if "light_Daylight" in df_bad.columns:
            df_bad.loc[0, "light_Daylight"] = np.nan
        # Run a minimal validation on invage instead (guaranteed to be imputed)
        df_bad2 = df.copy()
        df_bad2.loc[0, "invage"] = np.nan
        with pytest.raises(AssertionError):
            validate_dataset(df_bad2, feature_cols)

    def test_validation_fails_on_out_of_bounds_coordinates(self, valid_df_and_features):
        df, feature_cols = valid_df_and_features
        df_bad = df.copy()
        df_bad.loc[0, "latitude"] = 50.0   # Outside Toronto
        with pytest.raises(AssertionError):
            validate_dataset(df_bad, feature_cols)

    def test_validation_fails_on_non_binary_target(self, valid_df_and_features):
        df, feature_cols = valid_df_and_features
        df_bad = df.copy()
        df_bad.loc[0, "acclass_binary"] = 99
        with pytest.raises(AssertionError):
            validate_dataset(df_bad, feature_cols)

    def test_lat_lon_within_toronto_bounds(self, df_encoded):
        valid = df_encoded[
            df_encoded["latitude"].notna() & df_encoded["longitude"].notna()
        ]
        assert valid["latitude"].between(LAT_MIN, LAT_MAX).all(), \
            "Some latitude values outside Toronto bounds."
        assert valid["longitude"].between(LON_MIN, LON_MAX).all(), \
            "Some longitude values outside Toronto bounds."

    def test_temporal_features_are_numeric(self, df_encoded):
        for col in ["hour", "day_of_week", "month", "year", "is_weekend"]:
            assert pd.api.types.is_numeric_dtype(df_encoded[col]), \
                f"Temporal feature '{col}' is not numeric."


# ─────────────────────────────────────────────────────────────────────────────
# 10. Integration / smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_runs_without_error(self, tmp_path):
        """End-to-end pipeline must complete without raising exceptions."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from data_preparation import run_pipeline

        # Write synthetic data to a temp CSV
        df = _make_raw_df(300)
        csv_path = tmp_path / "test_ksi.csv"
        df.to_csv(csv_path, index=False)

        result = run_pipeline(str(csv_path), str(tmp_path / "outputs"))
        assert "df_encoded" in result
        assert "splits" in result
        assert "miss_report" in result

    def test_output_artefacts_created(self, tmp_path):
        """All expected output files must be written."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from data_preparation import run_pipeline

        df = _make_raw_df(300)
        csv_path = tmp_path / "test_ksi.csv"
        df.to_csv(csv_path, index=False)
        out_dir = tmp_path / "outputs"

        run_pipeline(str(csv_path), str(out_dir))

        expected_files = [
            "variable_catalogue.csv",
            "missingness_report.csv",
            "ksi_encoded.csv",
            "X_train_smote.csv",
            "y_train_smote.csv",
            "X_test.csv",
            "y_test.csv",
        ]
        for fname in expected_files:
            assert (out_dir / fname).exists(), f"Output file '{fname}' not created."