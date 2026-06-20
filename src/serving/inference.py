from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_PATH      = PROJECT_ROOT / "mlp_classification.h5"
XGB_PATH      = PROJECT_ROOT / "xgb_regression.json"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "preprocessing.pkl"

NUM_FEATURES = [
    "age",
    "admissionweight",
    "lab_workload_last_hour",
    "result_hour",
    "result_weekday",
    "result_year",
    "result_month",
    "result_day",
]
CAT_FEATURES = ["labname", "gender", "unittype", "recent_diagnosis"]
PERIOD_LABELS = ["apres_midi", "matin", "nuit", "soir"]

_MLP = None
_XGB = None
# XGBoost preprocessing
_ONEHOT = None
_SCALER = None
_ENCODER = None
_MEDIANS = None
# MLP preprocessing (separate scaler/encoder — different feature set)
_MLP_SCALER = None
_MLP_ENCODER = None
_MLP_MEDIANS = None


def _safe_int(value, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_result_datetime(row: pd.Series) -> pd.Timestamp:
    year = _safe_int(row.get("result_year"), 2026)
    month = min(max(_safe_int(row.get("result_month"), 1), 1), 12)
    day = min(max(_safe_int(row.get("result_day"), 1), 1), 31)
    hour = min(max(_safe_int(row.get("result_hour"), 0), 0), 23)

    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=hour)
    except ValueError:
        safe_day = min(day, pd.Period(f"{year}-{month:02d}").days_in_month)
        return pd.Timestamp(year=year, month=month, day=safe_day, hour=hour)



def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "result_hour" not in df.columns:
        df["result_hour"] = (pd.to_numeric(df.get("result_time"), errors="coerce") // 60) % 24

    if "result_weekday" not in df.columns:
        df["result_weekday"] = ((pd.to_numeric(df.get("result_time"), errors="coerce") // 1440) % 7).abs() % 7

    df["result_year"] = df.get("result_year", 2026)
    df["result_month"] = df.get("result_month", 1)
    df["result_day"] = df.get("result_day", 1)

    result_dates = df.apply(_safe_result_datetime, axis=1)
    missing_weekday = df["result_weekday"].isna() if "result_weekday" in df.columns else pd.Series(True, index=df.index)
    df.loc[missing_weekday, "result_weekday"] = result_dates.loc[missing_weekday].dt.weekday

    return df


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_time_features(df)

    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).str.strip()

    df["age"] = df["age"].replace({"> 89": 90})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["admissionweight"] = pd.to_numeric(df["admissionweight"], errors="coerce")

    for col in NUM_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    return df


_ARTIFACT = None   # full preprocessing dict, cached after first load


def _load_preprocessing() -> dict:
    """Load all preprocessing transformers saved by run_pipeline.py.

    Both the XGBoost and MLP have separate scalers/encoders because they
    use different feature sets — they must never be re-fitted at serving time.
    """
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing artifacts not found: {ARTIFACTS_PATH}\n"
            "Run the training pipeline first:\n"
            "  python scripts/run_pipeline.py --input lab_pred.csv"
        )
    return joblib.load(ARTIFACTS_PATH)


def load_models():
    global _MLP, _XGB, _ONEHOT, _SCALER, _ENCODER, _MEDIANS
    global _MLP_SCALER, _MLP_ENCODER, _MLP_MEDIANS

    if all(item is not None for item in [
        _MLP, _XGB, _ONEHOT, _SCALER, _ENCODER, _MEDIANS,
        _MLP_SCALER, _MLP_ENCODER, _MLP_MEDIANS,
    ]):
        return _MLP, _XGB, _ONEHOT, _SCALER, _ENCODER, _MEDIANS

    if tf is None:
        raise ImportError(
            "TensorFlow is required to load mlp_classification.h5.\n"
            "Install it with: pip install tensorflow"
        )
    if not MLP_PATH.exists():
        raise FileNotFoundError(f"MLP model not found: {MLP_PATH}")
    if not XGB_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {XGB_PATH}")

    _MLP = tf.keras.models.load_model(MLP_PATH)

    _XGB = XGBRegressor()
    _XGB.load_model(str(XGB_PATH))

    # Load all preprocessing artifacts saved during training
    global _ARTIFACT
    _ARTIFACT    = _load_preprocessing()
    _ONEHOT      = _ARTIFACT["onehot"]
    _SCALER      = _ARTIFACT["scaler"]
    _ENCODER     = _ARTIFACT["label_encoder"]
    _MEDIANS     = _ARTIFACT["medians"]
    _MLP_SCALER  = _ARTIFACT["mlp_scaler"]
    _MLP_ENCODER = _ARTIFACT["mlp_label_encoder"]
    _MLP_MEDIANS = _ARTIFACT["mlp_medians"]

    return _MLP, _XGB, _ONEHOT, _SCALER, _ENCODER, _MEDIANS


def _payload_to_frame(payload: dict) -> pd.DataFrame:
    row = {
        "labname": payload.get("labname"),
        "gender": payload.get("gender"),
        "age": payload.get("age"),
        "unittype": payload.get("unittype"),
        "recent_diagnosis": payload.get("recent_diagnosis") or "unknown",
        "result_time": payload.get("result_time"),
        "admissionweight": payload.get("admissionweight"),
        "lab_workload_last_hour": payload.get("lab_workload_last_hour"),
        "result_year": payload.get("result_year"),
        "result_month": payload.get("result_month"),
        "result_day": payload.get("result_day"),
        "result_hour": payload.get("result_hour"),
        "result_weekday": payload.get("result_weekday"),
    }
    return pd.DataFrame([row])


def _transform_payload(payload: dict):
    """Return (x_xgb, x_mlp, df) where:
    - x_xgb : sparse matrix  (NUM_FEATURES + OHE CAT_FEATURES) for XGBoost
    - x_mlp : dense ndarray  (MLP_FEATURES scaled)              for MLP
    - df    : cleaned single-row DataFrame (for datetime recovery)
    """
    load_models()  # ensure globals are populated

    df = _clean_features(_payload_to_frame(payload))

    # --- XGBoost input ---
    df[NUM_FEATURES] = df[NUM_FEATURES].fillna(_MEDIANS)
    x_num = _SCALER.transform(df[NUM_FEATURES].values)
    x_cat = _ONEHOT.transform(df[CAT_FEATURES])
    x_xgb = hstack([x_num, x_cat])

    # --- MLP input (5-feature subset, own scaler) ---
    mlp_feats  = _ARTIFACT["mlp_features"]
    x_mlp_raw  = df[mlp_feats].fillna(_MLP_MEDIANS).values.astype(np.float32)
    x_mlp      = _MLP_SCALER.transform(x_mlp_raw)

    return x_xgb, x_mlp, df


def predict(payload: dict) -> dict:
    try:
        mlp, xgb, _, _, _, _ = load_models()
        x_xgb, x_mlp, df = _transform_payload(payload)

        # --- Period classification (MLP, 5-feature scaled input) ---
        class_probs = mlp.predict(x_mlp, verbose=0)
        class_id    = int(np.argmax(class_probs, axis=1)[0])
        period      = _MLP_ENCODER.inverse_transform([class_id])[0]

        # --- Turnaround regression (XGBoost, full sparse input) ---
        turnaround = float(xgb.predict(x_xgb)[0])
        turnaround = max(0.0, turnaround)

        result_datetime     = _safe_result_datetime(df.iloc[0])
        validation_datetime = result_datetime + pd.to_timedelta(turnaround, unit="m")

        return {
            "status": "success",
            "predicted_turnaround_time_mins":  round(turnaround, 2),
            "predicted_period":                period,
            "predicted_validation_datetime":   validation_datetime.strftime("%d/%m/%Y %H:%M"),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}