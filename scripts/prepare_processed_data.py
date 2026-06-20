import os
import sys

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.data.preprocess import preprocess_data      # or wherever your modules live
from src.features.build_features import build_features

# === CONFIG ===
RAW = "data/raw/lab_pred.csv"
OUT = "data/processed/lab_pred_processed.csv"
TARGET_COL = "turnaround_time_mins"

# 1) load raw
print(f"Loading raw data from {RAW}...")
df = pd.read_csv(RAW)

# 2) preprocess (cleans text, fixes age, imputes missing values, handles outliers)
print("Preprocessing...")
df = preprocess_data(df, target_column=TARGET_COL)

# 3) sanity checks for target
assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' missing after preprocess"
assert df[TARGET_COL].isna().sum() == 0, f"Target '{TARGET_COL}' has NaNs after preprocess"
assert (df[TARGET_COL] > 0).all(), f"Target '{TARGET_COL}' has non-positive values"
assert df[TARGET_COL].dtype in ["int64", "float64"], f"Target '{TARGET_COL}' not numeric"

# 4) feature engineering (returns DataFrame + selected feature list)
print("Building features...")
df_processed, selected_features = build_features(df, target_column=TARGET_COL)

# 5) save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f"   Processed dataset saved to {OUT} | Shape: {df_processed.shape}")
print(f"   Selected features: {len(selected_features)}")
print(f"   Top 10 features: {selected_features[:10]}")