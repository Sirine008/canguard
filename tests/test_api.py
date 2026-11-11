"""API endpoint tests.

Endpoints that need trained artifacts or the knowledge base are skipped
automatically when those artifacts are absent, so the suite runs on a
fresh clone too.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import CHROMA_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

MODELS_READY = (MODELS_DIR / "lstm_ae.onnx").exists() and \
               (PROCESSED_DIR / "train_stats.json").exists()
KB_READY = CHROMA_DIR.exists()
METRICS_READY = (REPORTS_DIR / "metrics.json").exists()


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_versions" in body


def _window_payload(n: int = 64) -> dict:
    rng = np.random.default_rng(1)
    t = 1000.0
    messages = []
    for _ in range(n):
        t += float(rng.uniform(0.0004, 0.0006))
        messages.append({
            "timestamp": t,
            "can_id": int(rng.choice([0x130, 0x131, 0x316, 0x43F])),
            "dlc": 8,
            "data": [int(b) for b in rng.integers(0, 256, 8)],
        })
    return {"messages": messages, "model": "lstm_autoencoder"}


@pytest.mark.skipif(not MODELS_READY, reason="models not trained")
def test_detect_returns_score(client):
    response = client.post("/detect", json=_window_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["attack_type"] in ("normal", "dos", "fuzzy", "gear", "rpm")
    assert isinstance(body["is_attack"], bool)


@pytest.mark.skipif(not MODELS_READY, reason="models not trained")
def test_detect_rejects_wrong_window_size(client):
    payload = _window_payload(10)
    response = client.post("/detect", json=payload)
    assert response.status_code == 422


@pytest.mark.skipif(not MODELS_READY, reason="models not trained")
def test_detect_accepts_hex_ids(client):
    payload = _window_payload()
    for message in payload["messages"]:
        message["can_id"] = f"{message['can_id']:03x}"
    response = client.post("/detect", json=payload)
    assert response.status_code == 200


@pytest.mark.skipif(not METRICS_READY, reason="evaluation not run")
def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    detection = response.json()["detection"]
    assert set(detection["models"]) == {
        "isolation_forest", "lstm_autoencoder", "lightgbm"}


@pytest.mark.skipif(not KB_READY, reason="knowledge base not built")
def test_diagnose_known_code(client):
    response = client.post("/diagnose", json={"question": "What does P0420 mean?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    codes = [s["code"] for s in body["sources"] if s["code"]]
    assert "P0420" in codes


@pytest.mark.skipif(not KB_READY, reason="knowledge base not built")
def test_diagnose_refuses_out_of_scope(client):
    response = client.post("/diagnose", json={
        "question": "What is the best lasagna recipe?"})
    assert response.status_code == 200
    assert response.json()["grounded"] is False
