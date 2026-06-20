from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

try:
    import tensorflow as tf
except ImportError:
    tf = None


PROJECT_ROOT   = Path(__file__).resolve().parents[2]
MLP_PATH       = PROJECT_ROOT / "mlp_classification.h5"
XGB_PATH       = PROJECT_ROOT / "artifacts" / "model.pkl"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "preprocessing.pkl"

# Public constants — referenced by tests for mocking.
# NUM_FEATURES is the numeric feature set the XGBoost scaler was fitted on (5 cols).
# CAT_FEATURES are the raw categorical columns before one-hot encoding.
NUM_FEATURES = ["age", "admissionweight", "lab_workload_last_hour",
                "result_hour", "result_weekday"]
CAT_FEATURES = ["labname", "gender", "unittype", "recent_diagnosis"]

# Private aliases used internally
_RAW_CAT_FEATURES = CAT_FEATURES
_RAW_NUM_FEATURES = NUM_FEATURES

_MLP   = None
_XGB   = None
_ARTIFACT = None


def _safe_int(value, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_result_datetime(row: pd.Series) -> pd.Timestamp:
    year  = _safe_int(row.get("result_year"),  2026)
    month = min(max(_safe_int(row.get("result_month"), 1), 1), 12)
    day   = min(max(_safe_int(row.get("result_day"),   1), 1), 31)
    hour  = min(max(_safe_int(row.get("result_hour"),  0), 0), 23)
    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=hour)
    except ValueError:
        safe_day = min(day, pd.Period(f"{year}-{month:02d}").days_in_month)
        return pd.Timestamp(year=year, month=month, day=safe_day, hour=hour)


def _load_preprocessing() -> dict:
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing artifacts not found: {ARTIFACTS_PATH}\n"
            "Run the training pipeline first:\n"
            "  python scripts/run_pipeline.py --input lab_pred.csv"
        )
    return joblib.load(ARTIFACTS_PATH)


def load_models():
    global _MLP, _XGB, _ARTIFACT

    if _XGB is not None and _ARTIFACT is not None:
        return

    # Load preprocessing artifact
    _ARTIFACT = _load_preprocessing()

    # Load XGBoost
    if not XGB_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {XGB_PATH}")
    _XGB = joblib.load(str(XGB_PATH))

    # Load MLP (optional)
    if tf is not None and MLP_PATH.exists():
        _MLP = tf.keras.models.load_model(MLP_PATH)


def _payload_to_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "labname":               payload.get("labname"),
        "gender":                payload.get("gender"),
        "age":                   payload.get("age"),
        "unittype":              payload.get("unittype"),
        "recent_diagnosis":      payload.get("recent_diagnosis") or "unknown",
        "admissionweight":       payload.get("admissionweight"),
        "lab_workload_last_hour": payload.get("lab_workload_last_hour"),
        "result_hour":           payload.get("result_hour", 0),
        "result_weekday":        payload.get("result_weekday", 0),
        "result_year":           payload.get("result_year",  2026),
        "result_month":          payload.get("result_month", 1),
        "result_day":            payload.get("result_day",   1),
    }])


def _build_xgb_input(df: pd.DataFrame) -> np.ndarray:
    """
    Reconstruct the exact 30-feature vector the XGBoost model was trained on.
    We replicate what build_features() produced:
      - numeric cols scaled with the saved scaler
      - categorical cols one-hot encoded, aligned to saved feature_columns
    """
    art          = _ARTIFACT
    feature_cols = art["feature_columns"]   # the 30 selected features
    num_features = art["num_features"]      # ["age","admissionweight","lab_workload_last_hour","result_hour","result_weekday"]
    medians      = art["medians"]
    scaler       = art["scaler"]

    # ---- numeric ----
    df["age"]      = df["age"].replace({"> 89": 90})
    for col in num_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[num_features] = df[num_features].fillna(medians)

    # ---- categorical ----
    for col in _RAW_CAT_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    # Build a result DataFrame with all 30 feature columns, defaulting to 0
    out = pd.DataFrame(0, index=df.index, columns=feature_cols, dtype=np.float32)

    # Fill numeric features (scaled)
    scaled_num = scaler.transform(df[num_features].values)
    for i, col in enumerate(num_features):
        if col in out.columns:
            out[col] = scaled_num[:, i]

    # Fill one-hot categorical features by matching column names
    # build_features creates columns like "labname_bedside glucose", "unittype_micu" etc.
    for cat_col in _RAW_CAT_FEATURES:
        val = df[cat_col].iloc[0]
        ohe_col = f"{cat_col}_{val}"
        if ohe_col in out.columns:
            out[ohe_col] = 1.0

    # Fill any remaining boolean/binary engineered features that were in
    # feature_cols but not covered above — they stay 0 (safe default).

    return out.values


def _build_mlp_input(df: pd.DataFrame) -> np.ndarray:
    """Build the 5-feature MLP input using the MLP-specific scaler."""
    art         = _ARTIFACT
    mlp_feats   = art["mlp_features"]    # 5 numeric cols
    mlp_medians = art["mlp_medians"]
    mlp_scaler  = art["mlp_scaler"]

    X = df[mlp_feats].copy()
    for col in mlp_feats:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(mlp_medians)
    return mlp_scaler.transform(X.values.astype(np.float32))


def predict(payload: dict) -> dict:
    try:
        load_models()

        df = _payload_to_frame(payload)

        # --- XGBoost: turnaround regression ---
        x_xgb      = _build_xgb_input(df.copy())
        turnaround = float(_XGB.predict(x_xgb)[0])
        turnaround = max(0.0, turnaround)

        # --- MLP: period classification (if available) ---
        period = None
        if _MLP is not None:
            x_mlp      = _build_mlp_input(df.copy())
            class_probs = _MLP.predict(x_mlp, verbose=0)
            class_id    = int(np.argmax(class_probs, axis=1)[0])
            period      = _ARTIFACT["mlp_label_encoder"].inverse_transform([class_id])[0]

        result_datetime     = _safe_result_datetime(df.iloc[0])
        validation_datetime = result_datetime + pd.to_timedelta(turnaround, unit="m")

        return {
            "status":                          "success",
            "predicted_turnaround_time_mins":  round(turnaround, 2),
            "predicted_period":                period,
            "predicted_validation_datetime":   validation_datetime.strftime("%d/%m/%Y %H:%M"),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}