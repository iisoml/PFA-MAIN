"""
Tests for the prediction logic in src/serving/inference.py.

Imports inference.predict() directly — avoids importing src/app/app.py
which loads Gradio (heavy, slow, and not needed for unit tests).
For full integration tests against a live server, use the smoke test
in the CI workflow instead.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from serving.inference import predict

SAMPLE_PAYLOAD = {
    "labname": "potassium",
    "result_time": 360,
    "gender": "Female",
    "age": "78",
    "unittype": "Med-Surg ICU",
    "admissionweight": 65.5,
    "recent_diagnosis": "cardiovascular|ventricular disorders|hypertension",
    "lab_workload_last_hour": 25,
    "result_year": 2024,
    "result_month": 6,
    "result_day": 15,
    "result_hour": 6,
    "result_weekday": 5,
}

VALID_PERIODS = {"matin", "apres_midi", "soir", "nuit"}


def test_predict_returns_dict():
    result = predict(SAMPLE_PAYLOAD)
    assert isinstance(result, dict)


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


def test_predict_period_is_valid():
    result = predict(SAMPLE_PAYLOAD)
    assert result["predicted_period"] in VALID_PERIODS


def test_predict_datetime_format():
    """Validation datetime must be in dd/mm/yyyy HH:MM format."""
    from datetime import datetime
    result = predict(SAMPLE_PAYLOAD)
    dt_str = result["predicted_validation_datetime"]
    datetime.strptime(dt_str, "%d/%m/%Y %H:%M")  # raises if format is wrong


def test_predict_without_optional_recent_diagnosis():
    payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "recent_diagnosis"}
    result = predict(payload)
    assert result["status"] == "success"


def test_predict_empty_payload_returns_error_not_exception():
    """Garbage input must return {"status": "error"}, never raise."""
    result = predict({})
    assert result["status"] == "error"
    assert "message" in result