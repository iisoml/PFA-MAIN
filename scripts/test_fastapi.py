"""
Integration tests for the FastAPI prediction endpoint.
Uses FastAPI's TestClient — no running server required.
"""
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "labid": 12345678,
    "labname": "potassium",
    "result_time": 360,
    "validation_time": 400,
    "gender": "Female",
    "age": "78",
    "unittype": "Med-Surg ICU",
    "admissionweight": 65.5,
    "recent_diagnosis": "cardiovascular|ventricular disorders|hypertension",
    "lab_workload_last_hour": 25,
}


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_required_fields():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    body = response.json()
    assert body["status"] == "success"
    assert "predicted_turnaround_time_mins" in body
    assert "predicted_period" in body
    assert "predicted_validation_datetime" in body


def test_predict_turnaround_is_non_negative():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    body = response.json()
    assert body["predicted_turnaround_time_mins"] >= 0


def test_predict_period_is_valid():
    valid_periods = {"matin", "apres_midi", "soir", "nuit"}
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    body = response.json()
    assert body["predicted_period"] in valid_periods


def test_predict_missing_optional_fields_does_not_crash():
    """recent_diagnosis is optional — must not raise a 500."""
    payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "recent_diagnosis"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_bad_payload_returns_error_status():
    """Completely empty payload should return error status, not a 500."""
    response = client.post("/predict", json={})
    # Either a validation error (422) or our own {"status": "error"} response
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        assert response.json()["status"] == "error"