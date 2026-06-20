import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def preprocess_data(df: pd.DataFrame, target_column: str = "turnaround_time_mins") -> pd.DataFrame:
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and cleaning outliers.

    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.
    target_column (str): The target variable name.

    Returns:
    pd.DataFrame: A preprocessed DataFrame ready for feature engineering.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Validate required columns exist
    required_cols = [
        "labid", "labname", "result_time", "validation_time",
        target_column, "gender", "age", "unittype",
        "admissionweight", "recent_diagnosis", "lab_workload_last_hour"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep result_time and validation_time as integer minute offsets
    # (build_features.py extracts hour/weekday from these integers)
    # Do NOT convert to datetime here — that breaks downstream feature engineering

    # Clean text columns
    df["recent_diagnosis"] = df["recent_diagnosis"].str.lower().str.strip()
    df["labname"] = df["labname"].str.lower().str.strip()
    df["gender"] = df["gender"].str.lower().str.strip()
    df["unittype"] = df["unittype"].str.lower().str.strip()

    # Fill missing diagnosis
    df["recent_diagnosis"] = df["recent_diagnosis"].fillna("unknown")

    # Fix age: '> 89' → 90, then convert to numeric
    df["age"] = df["age"].replace({"> 89": 90})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Convert admissionweight to numeric
    df["admissionweight"] = pd.to_numeric(df["admissionweight"], errors="coerce")

    # Impute medians for numeric columns
    df["age"] = df["age"].fillna(df["age"].median())
    df["admissionweight"] = df["admissionweight"].fillna(df["admissionweight"].median())

    # --- Outlier handling for target ---
    # Turnaround times > 24 hours (1440 mins) are likely data errors
    # Cap at 99th percentile rather than dropping (preserves rows)
    turnaround_99th = df[target_column].quantile(0.99)
    df[target_column] = df[target_column].clip(upper=turnaround_99th)

    # Also ensure no negative or zero turnaround times
    df[target_column] = df[target_column].clip(lower=1)

    print(f"Preprocessing complete. Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Target '{target_column}' capped at 99th percentile: {turnaround_99th:.1f} mins")
    print(f"Target range after cleaning: [{df[target_column].min():.0f}, {df[target_column].max():.0f}]")

    return df


