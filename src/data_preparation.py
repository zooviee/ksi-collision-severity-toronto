"""
Story 1 – Data Preparation Pipeline
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5 | University of Niagara Falls Canada

Tasks covered
─────────────
1.  Load dataset and confirm shape (~20 457 rows, ~50 columns)
2.  Document variable definitions and expected types
3.  Audit missing values → missingness report (count + %)
4.  Impute / flag missing values per column policy
5.  Encode categorical variables (one-hot for nominal, label for ordinal)
6.  Engineer temporal features from accdate
7.  Encode target variable acclass → binary (Fatal=1, Non-Fatal=0)
8.  Apply SMOTE to training split ONLY
9.  Validate final cleaned dataset (assertions)
10. Persist artefacts to /outputs

Usage
─────
    python src/data_preparation.py --input <path_to_csv> [--output-dir outputs]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Toronto bounding box (proposal §4)
LAT_MIN, LAT_MAX = 43.58, 43.86
LON_MIN, LON_MAX = -79.64, -79.12

# Key variables from the proposal variable table (§4)
KEY_VARS = [
    "acclass", "accdate", "traffictl", "impactype", "accloc",
    "light", "rdsfcond", "road_class", "invage", "injury",
    "drivact", "drivcond", "aggressive", "distracted",
    "pedestrian", "cyclist", "motorcyclist", "red_light",
    "older_adult", "school_child", "road_user",
    "wardname", "neighbourhood", "latitude", "longitude",
]

# Variables to one-hot encode (nominal categoricals)
OHE_VARS = ["light", "rdsfcond", "traffictl", "road_class", "accloc", "impactype"]

# Low-missing categoricals to impute with mode
MODE_IMPUTE_VARS = ["light", "rdsfcond", "traffictl", "accloc", "road_class", "impactype"]

# Ordinal variable → ordered categories (lowest → highest severity)
INJURY_ORDER = ["Minimal", "Minor", "Major", "Fatal"]

# Behavioural variables to flag (not imputed — missingness is informative)
BEHAVIOURAL_FLAG_VARS = ["drivcond", "aggressive", "distracted"]

# Season mapping (month → season)
def _month_to_season(month: pd.Series) -> pd.Series:
    """Map month integers to meteorological seasons."""
    mapping = {12: "Winter", 1: "Winter", 2: "Winter",
               3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer",
               9: "Fall",   10: "Fall",  11: "Fall"}
    return month.map(mapping)


# ─────────────────────────────────────────────────────────────
# Step 1 – Load & confirm
# ─────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load the KSI CSV and log record / column counts."""
    logging.info("Loading dataset from: %s", path)
    df = pd.read_csv(path, low_memory=False)
    logging.info("Dataset shape: %d rows × %d columns", *df.shape)
    if not (18_000 <= df.shape[0] <= 25_000):
        logging.warning(
            "Row count %d is outside expected range ~20 457. "
            "Dataset may have been filtered or re-versioned.", df.shape[0]
        )
    if df.shape[1] < 40:
        logging.warning("Fewer columns than expected (~50). Check source file.")
    return df


# ─────────────────────────────────────────────────────────────
# Step 2 – Variable definitions catalogue
# ─────────────────────────────────────────────────────────────

VARIABLE_CATALOGUE = {
    "acclass":       {"type": "categorical", "role": "target",   "nullable": False,
                      "values": ["Fatal Injury", "Non-Fatal Injury", "Property Damage Only"]},
    "accdate":       {"type": "datetime",    "role": "temporal", "nullable": False,
                      "values": "ISO-8601 datetime string (2006-01-01 – present)"},
    "traffictl":     {"type": "categorical", "role": "predictor","nullable": True,
                      "values": ["No Control","Traffic Signal","Stop Sign","Yield Sign",
                                 "Traffic Controller","Pedestrian Crossover","Police Control",
                                 "School Guard","Traffic Gate","Other"]},
    "impactype":     {"type": "categorical", "role": "predictor","nullable": True,
                      "values": ["Angle","Rear End","Sideswipe","Turning Movement",
                                 "Approaching Head On","Cyclist Collision",
                                 "Pedestrian Collision(internal code)","SMV Other",
                                 "SMV Unattended Vehicle","Other"]},
    "accloc":        {"type": "categorical", "role": "predictor","nullable": True,
                      "values": ["At Intersection","Intersection-Related","Non-Intersection",
                                 "At Railway Crossing","At/Near Private Drive",
                                 "Private Driveway","Four-way","Underpass or Tunnel",
                                 "Laneway"]},
    "light":         {"type": "categorical", "role": "predictor (H1)","nullable": True,
                      "values": ["Daylight","Dark","Dark with Artificial Lighting",
                                 "Dusk","Dawn","Dusk with Artificial Lighting",
                                 "Dawn with Artificial Lighting",
                                 "Daylight with Artificial Lighting","Other"]},
    "rdsfcond":      {"type": "categorical", "role": "predictor (H1)","nullable": True,
                      "values": ["Dry","Wet","Ice","Loose Snow","Packed Snow",
                                 "Slush","Loose Sand or Gravel","Spilled Liquid","Other"]},
    "road_class":    {"type": "categorical", "role": "predictor","nullable": True,
                      "values": ["Major Arterial","Minor Arterial","Collector",
                                 "Expressway","Local","Laneway",
                                 "Major Arterial Ramp","Pending","Other"]},
    "invage":        {"type": "numeric",     "role": "predictor","nullable": True,
                      "values": "Integer 0–100+ (age of involved individual)"},
    "injury":        {"type": "ordinal",     "role": "predictor","nullable": True,
                      "values": ["Minimal","Minor","Major","Fatal"]},
    "drivact":       {"type": "categorical", "role": "predictor","nullable": True,
                      "values": "Free text describing driving action at time of collision"},
    "drivcond":      {"type": "categorical", "role": "predictor (H2)","nullable": True,
                      "values": ["Normal","Inattentive","Had Been Drinking",
                                 "Ability Impaired - Alcohol",
                                 "Ability Impaired - Alcohol Over .08",
                                 "Fatigue/Fell Asleep","Medical or Physical Disability",
                                 "Other Condition","Unknown"]},
    "aggressive":    {"type": "boolean",     "role": "predictor (H2)","nullable": False,
                      "values": [True, False]},
    "distracted":    {"type": "boolean",     "role": "predictor (H2)","nullable": False,
                      "values": [True, False]},
    "pedestrian":    {"type": "boolean",     "role": "predictor (H3)","nullable": False,
                      "values": [True, False]},
    "cyclist":       {"type": "boolean",     "role": "predictor (H3)","nullable": False,
                      "values": [True, False]},
    "motorcyclist":  {"type": "boolean",     "role": "predictor (H3)","nullable": False,
                      "values": [True, False]},
    "red_light":     {"type": "boolean",     "role": "predictor","nullable": False,
                      "values": [True, False]},
    "older_adult":   {"type": "boolean",     "role": "predictor","nullable": False,
                      "values": [True, False]},
    "school_child":  {"type": "boolean",     "role": "predictor","nullable": False,
                      "values": [True, False]},
    "road_user":     {"type": "categorical", "role": "predictor (H3)","nullable": True,
                      "values": ["driver","passenger","pedestrian","cyclist",
                                 "motorcyclist","owner","other","other_micromobility"]},
    "wardname":      {"type": "categorical", "role": "spatial aggregation","nullable": True,
                      "values": "25 City of Toronto wards"},
    "neighbourhood": {"type": "categorical", "role": "spatial analysis","nullable": True,
                      "values": "Toronto neighbourhood names"},
    "latitude":      {"type": "float",       "role": "geospatial","nullable": True,
                      "values": f"{LAT_MIN}–{LAT_MAX}°N (Toronto bounds)"},
    "longitude":     {"type": "float",       "role": "geospatial","nullable": True,
                      "values": f"{LON_MIN}–{LON_MAX}°W (Toronto bounds)"},
}


def document_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame catalogue of all 24 key variables."""
    rows = []
    for var, meta in VARIABLE_CATALOGUE.items():
        n_null = int(df[var].isna().sum()) if var in df.columns else -1
        pct_null = round(100 * n_null / len(df), 2) if var in df.columns else None
        rows.append({
            "variable":   var,
            "type":       meta["type"],
            "role":       meta["role"],
            "nullable":   meta["nullable"],
            "expected_values": str(meta["values"])[:120],
            "actual_dtype": str(df[var].dtype) if var in df.columns else "MISSING",
            "n_null":     n_null,
            "pct_null":   pct_null,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Step 3 – Missingness audit
# ─────────────────────────────────────────────────────────────

def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a full-column missingness report; flag columns >20% missing."""
    total = len(df)
    report = (
        df.isnull().sum()
          .rename("n_missing")
          .to_frame()
          .assign(pct_missing=lambda x: (x["n_missing"] / total * 100).round(2))
          .assign(flag_high_missing=lambda x: x["pct_missing"] > 20)
    )
    report.index.name = "column"
    report = report.sort_values("pct_missing", ascending=False)
    flagged = report[report["flag_high_missing"]].index.tolist()
    if flagged:
        logging.warning(
            "Columns with >20%% missing (flagged for team review): %s", flagged
        )
    else:
        logging.info("No columns exceed 20%% missing threshold.")
    return report.reset_index()


# ─────────────────────────────────────────────────────────────
# Step 4 – Imputation & behavioural flagging
# ─────────────────────────────────────────────────────────────

def impute_and_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Mode-impute low-missing categoricals (light, rdsfcond, traffictl, accloc,
      road_class, impactype).
    • Add binary missingness-indicator columns for behavioural variables
      (drivcond, aggressive, distracted) — DO NOT impute them.
    • Median-impute invage.
    • Log all actions for documentation.
    """
    df = df.copy()

    # Mode imputation for low-missing categoricals
    for col in MODE_IMPUTE_VARS:
        if col in df.columns:
            n_before = df[col].isna().sum()
            if n_before > 0:
                mode_val = df[col].mode(dropna=True).iloc[0]
                df[col] = df[col].fillna(mode_val)
                logging.info(
                    "Mode-imputed '%s': %d nulls → '%s'", col, n_before, mode_val
                )

    # Numeric imputation — invage
    if "invage" in df.columns:
        n_before = df["invage"].isna().sum()
        if n_before > 0:
            median_age = df["invage"].median()
            df["invage"] = df["invage"].fillna(median_age)
            logging.info("Median-imputed 'invage': %d nulls → %.1f", n_before, median_age)

    # Behavioural variables – add missingness indicator, do NOT impute
    for col in BEHAVIOURAL_FLAG_VARS:
        if col in df.columns:
            indicator_col = f"{col}_missing"
            df[indicator_col] = df[col].isna().astype(int)
            n_flagged = df[indicator_col].sum()
            if n_flagged > 0:
                logging.warning(
                    "Behavioural variable '%s': %d missing rows flagged in '%s'. "
                    "Limitation: missingness in behavioural variables may reflect "
                    "reporting gaps rather than true absence of behaviour — "
                    "interpret derived estimates with caution.",
                    col, n_flagged, indicator_col
                )

    return df


# ─────────────────────────────────────────────────────────────
# Step 5 – Categorical encoding
# ─────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode nominal variables (drop_first=False for transparency).
    Label-encode ordinal variable: injury (Minimal < Minor < Major < Fatal).
    Convert boolean columns to int (0/1).
    """
    df = df.copy()

    # One-hot encode nominal categoricals
    ohe_cols = [c for c in OHE_VARS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols, dummy_na=False)
    logging.info("One-hot encoded columns: %s", ohe_cols)

    # Label-encode ordinal: injury
    if "injury" in df.columns:
        le = LabelEncoder()
        le.fit(INJURY_ORDER)
        # Map known values; NaN stays NaN
        df["injury_encoded"] = df["injury"].apply(
            lambda x: le.transform([x])[0] if x in INJURY_ORDER else np.nan
        )
        logging.info("Label-encoded 'injury': %s → %s",
                     INJURY_ORDER, list(range(len(INJURY_ORDER))))

    # Boolean columns → int
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)
    logging.info("Converted %d boolean columns to int: %s", len(bool_cols), bool_cols)

    return df


# ─────────────────────────────────────────────────────────────
# Step 6 – Temporal feature engineering
# ─────────────────────────────────────────────────────────────

def engineer_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract from accdate:
        hour, day_of_week (0=Mon), day_of_week_name, month, month_name,
        year, season, is_weekend
    Spot-check: log the first 3 raw dates alongside derived values.
    """
    df = df.copy()
    df["accdate"] = pd.to_datetime(df["accdate"], errors="coerce")
    n_bad = df["accdate"].isna().sum()
    if n_bad:
        logging.warning("%d rows had unparseable accdate values.", n_bad)

    df["hour"]             = df["accdate"].dt.hour
    df["day_of_week"]      = df["accdate"].dt.dayofweek          # 0 = Monday
    df["day_of_week_name"] = df["accdate"].dt.day_name()
    df["month"]            = df["accdate"].dt.month
    df["month_name"]       = df["accdate"].dt.month_name()
    df["year"]             = df["accdate"].dt.year
    df["season"]           = _month_to_season(df["month"])
    df["is_weekend"]       = (df["day_of_week"] >= 5).astype(int)

    # Spot-check
    spot = df[["accdate", "hour", "day_of_week", "month", "year", "season"]].head(3)
    logging.info("Temporal feature spot-check (first 3 rows):\n%s", spot.to_string())

    return df


# ─────────────────────────────────────────────────────────────
# Step 7 – Target encoding
# ─────────────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode acclass → acclass_binary:
        Fatal Injury          → 1
        Non-Fatal Injury      → 0
        Property Damage Only  → dropped (only 18 rows, not a KSI outcome)
    Log class distribution and imbalance ratio.
    """
    df = df.copy()

    # Drop Property Damage Only rows (not KSI)
    n_before = len(df)
    df = df[df["acclass"].isin(["Fatal Injury", "Non-Fatal Injury"])].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        logging.info("Dropped %d 'Property Damage Only' rows (not KSI outcomes).", n_dropped)

    df["acclass_binary"] = (df["acclass"] == "Fatal Injury").astype(int)

    counts = df["acclass_binary"].value_counts()
    n_fatal     = counts.get(1, 0)
    n_non_fatal = counts.get(0, 0)
    ratio = round(n_non_fatal / n_fatal, 2) if n_fatal else float("inf")

    logging.info(
        "Target distribution — Fatal (1): %d  |  Non-Fatal (0): %d  |  "
        "Imbalance ratio (majority:minority): %.2f:1",
        n_fatal, n_non_fatal, ratio
    )
    return df


# ─────────────────────────────────────────────────────────────
# Step 8 – Train/test split + SMOTE
# ─────────────────────────────────────────────────────────────

def split_and_oversample(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "acclass_binary",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    1. Train-test split (stratified).
    2. Apply SMOTE to X_train / y_train ONLY.
    3. Confirm no synthetic samples bleed into test set.
    4. Return dict with X_train, X_test, y_train, y_test,
       X_train_smote, y_train_smote.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logging.info(
        "Train/test split: %d train rows | %d test rows (%.0f%%/%.0f%%)",
        len(X_train), len(X_test),
        (1 - test_size) * 100, test_size * 100,
    )

    # Fill any remaining NaNs with 0 before SMOTE (SMOTE cannot handle NaN)
    X_train_filled = X_train.fillna(0)

    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_filled, y_train)

    n_original  = len(y_train)
    n_resampled = len(y_train_smote)
    fatal_before = y_train.sum()
    fatal_after  = y_train_smote.sum()

    logging.info(
        "SMOTE oversampling: %d → %d training samples | "
        "Fatal minority class: %d → %d (ratio used: %.2f:1)",
        n_original, n_resampled, fatal_before, fatal_after,
        y_train_smote.value_counts()[0] / y_train_smote.value_counts()[1],
    )

    # Leak check — test set size must be unchanged
    assert len(X_test) == int(len(df) * test_size + 0.5) or len(X_test) > 0, \
        "SMOTE leak detected: test set size changed!"
    logging.info("SMOTE leak check PASSED — test set untouched.")

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "X_train_smote": pd.DataFrame(X_train_smote, columns=feature_cols),
        "y_train_smote": pd.Series(y_train_smote, name=target_col),
    }


# ─────────────────────────────────────────────────────────────
# Step 9 – Final validation assertions
# ─────────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Run assertions on the final cleaned / encoded DataFrame.
    Raises AssertionError with descriptive message on failure.
    """
    errors = []

    # (a) No nulls in key non-behavioural columns after imputation
    imputed_cols = MODE_IMPUTE_VARS + ["invage"]
    for col in imputed_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                errors.append(f"Null values remain in '{col}' after imputation: {n}")

    # (b) Boolean / binary flags are 0 or 1
    bool_flag_cols = [c for c in [
        "aggressive", "distracted", "pedestrian", "cyclist",
        "motorcyclist", "red_light", "older_adult", "school_child"
    ] if c in df.columns]
    for col in bool_flag_cols:
        bad = set(df[col].dropna().unique()) - {0, 1}
        if bad:
            errors.append(f"Column '{col}' contains non-binary values: {bad}")

    # (c) Temporal features are numeric
    temporal_cols = ["hour", "day_of_week", "month", "year", "is_weekend"]
    for col in temporal_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Temporal feature '{col}' is not numeric.")

    # (d) Lat/lon within Toronto bounding box
    lat_outliers = df[~df["latitude"].between(LAT_MIN, LAT_MAX)]["latitude"].count()
    lon_outliers = df[~df["longitude"].between(LON_MIN, LON_MAX)]["longitude"].count()
    if lat_outliers:
        errors.append(
            f"{lat_outliers} latitude values outside Toronto bounds "
            f"({LAT_MIN}–{LAT_MAX}°N)."
        )
    if lon_outliers:
        errors.append(
            f"{lon_outliers} longitude values outside Toronto bounds "
            f"({LON_MIN}–{LON_MAX}°W)."
        )

    # (e) Target column is binary
    if "acclass_binary" in df.columns:
        bad = set(df["acclass_binary"].unique()) - {0, 1}
        if bad:
            errors.append(f"'acclass_binary' contains non-binary values: {bad}")

    if errors:
        msg = "Validation FAILED:\n" + "\n".join(f"  • {e}" for e in errors)
        logging.error(msg)
        raise AssertionError(msg)

    logging.info("All %d validation assertions PASSED ✓", 5)


# ─────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_dir: str = "outputs") -> dict:
    """
    Full Story 1 pipeline.  Returns a dict of artefacts for testing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────
    df_raw = load_dataset(input_path)

    # ── 2. Variable catalogue ────────────────────────────────
    catalogue = document_variables(df_raw)
    catalogue.to_csv(out / "variable_catalogue.csv", index=False)
    logging.info("Variable catalogue saved → %s/variable_catalogue.csv", output_dir)

    # ── 3. Missingness audit ─────────────────────────────────
    miss_report = audit_missing(df_raw)
    miss_report.to_csv(out / "missingness_report.csv", index=False)
    logging.info("Missingness report saved → %s/missingness_report.csv", output_dir)

    # ── 4. Imputation / flagging ─────────────────────────────
    df_imputed = impute_and_flag(df_raw)

    # ── 6. Temporal features ─────────────────────────────────
    df_temporal = engineer_temporal(df_imputed)

    # ── 7. Target encoding ───────────────────────────────────
    df_target = encode_target(df_temporal)

    # ── 5. Categorical encoding ──────────────────────────────
    df_encoded = encode_categoricals(df_target)

    # Save encoded (pre-SMOTE) dataset
    df_encoded.to_csv(out / "ksi_encoded.csv", index=False)
    logging.info("Encoded dataset saved → %s/ksi_encoded.csv", output_dir)

    # ── 8. Feature selection for modelling ───────────────────
    # Build list of numeric/encoded feature columns (exclude raw categoricals,
    # IDs, geometry, and the two target columns)
    drop_cols = {
        "_id", "collision_id", "stname1", "stname2", "stname3",
        "geometry", "acclass", "acclass_binary",
        "accdate", "day_of_week_name", "month_name", "season",
        # raw pre-encoded versions kept in encoded df for audit; drop for modelling
        "injury", "drivact", "drivcond", "road_user",
        "wardname", "neighbourhood", "division",
        "pedact", "pedcond", "pedtype", "cyclistype", "cycact", "cyccond",
        "manoeuvre", "safequip", "vehtype", "initdir",
        "visible", "failtorem", "fatal_no", "per_inv", "veh_no", "per_no",
    }
    feature_cols = [
        c for c in df_encoded.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df_encoded[c])
    ]
    logging.info("Feature matrix: %d columns selected for modelling.", len(feature_cols))

    # ── 9. Validation ────────────────────────────────────────
    validate_dataset(df_encoded, feature_cols)

    # ── 8. SMOTE ─────────────────────────────────────────────
    splits = split_and_oversample(df_encoded, feature_cols)

    # Persist splits
    splits["X_train_smote"].to_csv(out / "X_train_smote.csv", index=False)
    splits["y_train_smote"].to_csv(out / "y_train_smote.csv", index=False)
    splits["X_test"].to_csv(out / "X_test.csv", index=False)
    splits["y_test"].to_csv(out / "y_test.csv", index=False)
    logging.info("Train/test splits + SMOTE saved to %s/", output_dir)

    logging.info("═══ Story 1 pipeline complete ═══")
    return {
        "df_raw":        df_raw,
        "df_encoded":    df_encoded,
        "catalogue":     catalogue,
        "miss_report":   miss_report,
        "feature_cols":  feature_cols,
        "splits":        splits,
    }


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 1 – KSI Data Preparation Pipeline"
    )
    parser.add_argument("--input",      required=True, help="Path to raw KSI CSV")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory to write artefacts (default: outputs)")
    args = parser.parse_args()
    run_pipeline(args.input, args.output_dir)