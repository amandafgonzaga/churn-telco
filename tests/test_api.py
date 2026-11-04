from fastapi.testclient import TestClient

from churn.api.main import app


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_predict_returns_probability(client, customer_payload):
    response = client.post("/predict", json=customer_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["churn_prediction"], bool)


def test_predict_accepts_missing_total_charges(client, customer_payload):
    customer_payload.pop("TotalCharges")

    assert client.post("/predict", json=customer_payload).status_code == 200


def test_predict_rejects_invalid_category(client, customer_payload):
    customer_payload["Contract"] = "Weekly"

    assert client.post("/predict", json=customer_payload).status_code == 422


def test_predict_batch(client, customer_payload):
    response = client.post("/predict/batch", json=[customer_payload, customer_payload])

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_predict_without_model_returns_503(customer_payload, monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(tmp_path / "inexistente.joblib"))
    app.state.model = None
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=customer_payload).status_code == 503
