import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def build_features(df: pd.DataFrame, target_column: str = "turnaround_time_mins") -> tuple[pd.DataFrame, list]:
    """
    Build features for the given DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.
    target_column (str): The target variable name.

    Returns:
    tuple: (pd.DataFrame with engineered features, list of selected feature names)
    """
    df = df.copy()

    # --- Temporal features from result_time (minutes offset) ---
    # Convert to hour of day (0-23)
    df["result_hour"] = (df["result_time"] // 60) % 24
    
    # Extract day of week (0=Monday, 6=Sunday) from minutes offset
    # Assuming minute 0 is a known reference point
    df["result_weekday"] = ((df["result_time"] // 1440) % 7).abs() % 7

    # Time of day category
    def categorize_hour(hour):
        if 6 <= hour < 12:    return "matin"
        elif 12 <= hour < 18: return "apres_midi"
        elif 18 <= hour < 24: return "soir"
        else:                  return "nuit"

    # Day category
    def categorize_weekday(day):
        return "weekend" if day in [5, 6] else "weekday"

    # Workload category
    def categorize_workload(w):
        if w < 10:    return "faible"
        elif w < 30:  return "moyen"
        else:          return "eleve"

    df["time_category"]     = df["result_hour"].apply(categorize_hour)
    df["day_category"]      = df["result_weekday"].apply(categorize_weekday)
    df["workload_category"] = df["lab_workload_last_hour"].apply(categorize_workload)

    # Drop raw time columns and identifier columns (not features)
    df = df.drop(columns=["result_time", "validation_time", "labid"])

    # --- Handle missing values BEFORE encoding ---
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include="number").columns:
        if col != target_column:
            df[col] = df[col].fillna(df[col].median())

    # One-hot encoding (drop_first to avoid collinearity)
    df_encoded = pd.get_dummies(df, drop_first=True)

    # --- Feature selection via Random Forest importance ---
    feature_cols = [c for c in df_encoded.columns if c != target_column]
    X_all = df_encoded[feature_cols]
    y_all = df_encoded[target_column]

    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_all, y_all)

    importances = pd.Series(rf_selector.feature_importances_, index=X_all.columns)
    
    # Select top N features by importance (or use a threshold)
    top_n = min(30, len(importances))
    selected_features = importances.nlargest(top_n).index.tolist()

    print(f"Feature engineering completed. Total features: {len(feature_cols)}, Selected: {len(selected_features)}")
    print(f"Top 10 features: {selected_features[:10]}")

    return df_encoded, selected_features