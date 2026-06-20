"""
Unit tests for src/serving/inference.py

All model files (mlp_classification.h5, xgb_regression.json, preprocessing.pkl)
are mocked — no training run required in CI.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from scipy.sparse import csr_matrix

import serving.inference as inference_module
from serving.inference import predict

# ---------------------------------------------------------------------------
# Fake model objects
# ---------------------------------------------------------------------------

def _make_fake_mlp():
    """MLP that returns class 1 ('apres_midi') for any input."""
    mlp = MagicMock()
    mlp.predict.return_value = np.array([[0.1, 0.7, 0.1, 0.1]])
    return mlp


def _make_fake_xgb():
    """XGBoost that always predicts 45.0 minutes."""
    xgb = MagicMock()
    xgb.predict.return_value = np.array([45.0])
    return xgb


def _make_fake_preprocessing():
    """Return (onehot, scaler, label_encoder, medians, mlp_scaler, mlp_encoder, mlp_medians, artifact)."""
    n_cat_cols = 10

    onehot = MagicMock()
    onehot.transform.return_value = csr_matrix(np.zeros((1, n_cat_cols)))

    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, len(inference_module.NUM_FEATURES)))

    label_encoder = MagicMock()
    label_encoder.inverse_transform.return_value = ["apres_midi"]

    medians = pd.Series(
        {col: 0.0 for col in inference_module.NUM_FEATURES}
    )

    mlp_scaler = MagicMock()
    mlp_scaler.transform.return_value = np.zeros((1, 5))

    mlp_encoder = MagicMock()
    mlp_encoder.inverse_transform.return_value = ["apres_midi"]

    mlp_medians = pd.Series({col: 0.0 for col in ["age", "admissionweight",
                              "lab_workload_last_hour", "result_hour", "result_weekday"]})

    artifact = {
        "onehot":          onehot,
        "scaler":          scaler,
        "label_encoder":   label_encoder,
        "medians":         medians,
        "mlp_features":    ["age", "admissionweight", "lab_workload_last_hour",
                            "result_hour", "result_weekday"],
        "mlp_scaler":      mlp_scaler,
        "mlp_label_encoder": mlp_encoder,
        "mlp_medians":     mlp_medians,
    }
    return onehot, scaler, label_encoder, medians, mlp_scaler, mlp_encoder, mlp_medians, artifact


# ---------------------------------------------------------------------------
# Fixture: inject fake models into the module globals before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_models():
    """Patch module-level globals so load_models() returns immediately."""
    onehot, scaler, label_encoder, medians, mlp_scaler, mlp_encoder, mlp_medians, artifact = (
        _make_fake_preprocessing()
    )

    with patch.multiple(
        "serving.inference",
        _MLP=_make_fake_mlp(),
        _XGB=_make_fake_xgb(),
        _ONEHOT=onehot,
        _SCALER=scaler,
        _ENCODER=label_encoder,
        _MEDIANS=medians,
        _MLP_SCALER=mlp_scaler,
        _MLP_ENCODER=mlp_encoder,
        _MLP_MEDIANS=mlp_medians,
        _ARTIFACT=artifact,
    ):
        yield


# ---------------------------------------------------------------------------
# Sample payload
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD = {
    "labname":                 "potassium",
    "result_time":             360,
    "gender":                  "Female",
    "age":                     "78",
    "unittype":                "Med-Surg ICU",
    "admissionweight":         65.5,
    "recent_diagnosis":        "cardiovascular|hypertension",
    "lab_workload_last_hour":  25,
    "result_year":             2024,
    "result_month":            6,
    "result_day":              15,
    "result_hour":             6,
    "result_weekday":          5,
}

VALID_PERIODS = {"matin", "apres_midi", "soir", "nuit"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_predict_returns_dict():
    assert isinstance(predict(SAMPLE_PAYLOAD), dict)


def test_predict_status_is_success():
    result = predict(SAMPLE_PAYLOAD)
    assert result["status"] == "success", f"Unexpected error: {result.get('message')}"


def test_predict_has_required_fields():
    result = predict(SAMPLE_PAYLOAD)
    assert "predicted_turnaround_time_mins" in result
    assert "predicted_period" in result
    assert "predicted_validation_datetime" in result


def test_predict_turnaround_is_non_negative():
    result = predict(SAMPLE_PAYLOAD)
    assert result["predicted_turnaround_time_mins"] >= 0


def test_predict_turnaround_matches_fake_model():
    result = predict(SAMPLE_PAYLOAD)
    assert result["predicted_turnaround_time_mins"] == 45.0


def test_predict_period_is_valid():
    result = predict(SAMPLE_PAYLOAD)
    assert result["predicted_period"] in VALID_PERIODS


def test_predict_datetime_format():
    from datetime import datetime
    result = predict(SAMPLE_PAYLOAD)
    datetime.strptime(result["predicted_validation_datetime"], "%d/%m/%Y %H:%M")


def test_predict_without_optional_recent_diagnosis():
    payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "recent_diagnosis"}
    result = predict(payload)
    assert result["status"] == "success"


def test_predict_empty_payload_returns_error_not_exception():
    result = predict({})
    # Must return a dict with status key — never raise
    assert isinstance(result, dict)
    assert "status" in result