import os
import pandas as pd

# Make sure Python can find your modules
import sys
sys.path.append(os.path.abspath("."))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

# === CONFIG ===
DATA_PATH = "lab_pred.csv"  # adjust to your file path
TARGET_COL = "turnaround_time_mins"

def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")

    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_column=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))

    # 3. Build Features
    print("\n[3] Building features...")
    df_features, selected_features = build_features(df_clean, target_column=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(f"Selected features ({len(selected_features)}): {selected_features[:10]}...")
    print(df_features.head(3))

    print("\n Phase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main()