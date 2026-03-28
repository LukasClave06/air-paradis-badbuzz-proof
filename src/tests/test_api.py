import os
import pytest

os.environ["UNIT_TEST_MODE"] = "1"

from src.api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.get_json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "unit_test_mode" in data
    assert data["unit_test_mode"] is True


def test_predict_positive(client):
    response = client.post(
        "/predict",
        json={"text": "Air Paradis is amazing, best flight ever!"}
    )

    assert response.status_code == 200
    data = response.get_json()

    assert "proba_pos" in data
    assert "proba_neg" in data
    assert "pred_label" in data
    assert "bad_buzz" in data
    assert "tweet_clean" in data


def test_predict_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data