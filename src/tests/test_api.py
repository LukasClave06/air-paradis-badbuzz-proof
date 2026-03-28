import os
import pytest

# IMPORTANT : activer le mode test AVANT import de l'app
os.environ["UNIT_TEST_MODE"] = "1"

from src.api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# -----------------------------
# HEALTH CHECK
# -----------------------------
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.get_json()

    assert "status" in data
    assert data["status"] == "ok"

    # Vérifie qu'on est bien en mode test
    assert "unit_test_mode" in data
    assert data["unit_test_mode"] is True

    # Vérifie infos supplémentaires utiles
    assert "model_type" in data
    assert "model_path" in data


# -----------------------------
# HOME PAGE
# -----------------------------
def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200


# -----------------------------
# PREDICT - CAS VALIDE
# -----------------------------
def test_predict_positive(client):
    response = client.post(
        "/predict",
        json={"text": "Air Paradis is amazing, best flight ever!"}
    )

    assert response.status_code == 200
    data = response.get_json()

    # Vérifie structure de sortie
    assert "tweet" in data
    assert "tweet_clean" in data
    assert "proba_pos" in data
    assert "proba_neg" in data
    assert "pred_label" in data
    assert "pred_text" in data
    assert "threshold" in data
    assert "bad_buzz" in data

    # Vérifie types
    assert isinstance(data["proba_pos"], float)
    assert isinstance(data["proba_neg"], float)
    assert isinstance(data["pred_label"], int)
    assert isinstance(data["bad_buzz"], bool)


# -----------------------------
# PREDICT - TEXTE VIDE
# -----------------------------
def test_predict_missing_text(client):
    response = client.post("/predict", json={})

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


# -----------------------------
# PREDICT - TEXTE VIDE STRING
# -----------------------------
def test_predict_empty_string(client):
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


# -----------------------------
# PREDICT - TYPE INVALID
# -----------------------------
def test_predict_invalid_type(client):
    response = client.post("/predict", json={"text": 12345})

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data