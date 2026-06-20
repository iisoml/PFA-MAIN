"""
Unit tests for Phase 1 pipeline: load → preprocess → build_features.
Runs entirely in-memory with a small synthetic DataFrame — no CSV file needed.
"""
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
import pandas as pd
import numpy as np

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

TARGET_COL = "turnaround_time_mins"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_df(n=50) -> pd.DataFrame:
    """Return a minimal synthetic DataFrame matching lab_pred.csv schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "labid":                   rng.integers(1000, 9999, n),
        "labname":                 rng.choice(["potassium", "glucose", "sodium"], n),
        "result_time":             rng.integers(0, 1440, n).astype(float),
        "validation_time":         rng.integers(10, 1500, n).astype(float),
        TARGET_COL:                rng.integers(5, 300, n).astype(float),
        "gender":                  rng.choice(["Male", "Female"], n),
        "age":                     rng.integers(18, 90, n).astype(str),
        "unittype":                rng.choice(["Med-Surg ICU", "CSICU"], n),
        "admissionweight":         rng.uniform(40, 120, n),
        "recent_diagnosis":        rng.choice(["hypertension", "unknown"], n),
        "lab_workload_last_hour":  rng.integers(0, 50, n).astype(float),
    })


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_returns_dataframe(tmp_path):
    df = _make_sample_df()
    csv_path = tmp_path / "lab.csv"
    df.to_csv(csv_path, index=False)
    result = load_data(str(csv_path))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_load_data_preserves_columns(tmp_path):
    df = _make_sample_df()
    csv_path = tmp_path / "lab.csv"
    df.to_csv(csv_path, index=False)
    result = load_data(str(csv_path))
    assert set(df.columns).issubset(set(result.columns))


# ---------------------------------------------------------------------------
# preprocess_data
# ---------------------------------------------------------------------------

def test_preprocess_drops_no_rows_on_clean_data():
    df = _make_sample_df()
    result = preprocess_data(df.copy(), target_column=TARGET_COL)
    # Rows may be filtered but the result must be a non-empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_preprocess_target_has_no_nulls():
    df = _make_sample_df()
    result = preprocess_data(df.copy(), target_column=TARGET_COL)
    assert result[TARGET_COL].isna().sum() == 0


def test_preprocess_target_is_positive():
    df = _make_sample_df()
    result = preprocess_data(df.copy(), target_column=TARGET_COL)
    assert (result[TARGET_COL] > 0).all()


def test_preprocess_handles_age_over_89():
    df = _make_sample_df(n=10)
    df.loc[0, "age"] = "> 89"
    result = preprocess_data(df.copy(), target_column=TARGET_COL)
    assert pd.to_numeric(result["age"], errors="coerce").notna().all()


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

def test_build_features_returns_tuple():
    df = _make_sample_df()
    df = preprocess_data(df, target_column=TARGET_COL)
    result = build_features(df, target_column=TARGET_COL)
    assert isinstance(result, tuple) and len(result) == 2


def test_build_features_selected_features_excludes_target():
    df = _make_sample_df()
    df = preprocess_data(df, target_column=TARGET_COL)
    _, selected = build_features(df, target_column=TARGET_COL)
    assert TARGET_COL not in selected


def test_build_features_returns_nonempty_feature_list():
    df = _make_sample_df()
    df = preprocess_data(df, target_column=TARGET_COL)
    _, selected = build_features(df, target_column=TARGET_COL)
    assert len(selected) > 0


def test_build_features_output_has_selected_columns():
    df = _make_sample_df()
    df = preprocess_data(df, target_column=TARGET_COL)
    df_enc, selected = build_features(df, target_column=TARGET_COL)
    for col in selected:
        assert col in df_enc.columns