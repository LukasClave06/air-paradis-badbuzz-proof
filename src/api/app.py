import os
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(
        BASE_DIR,
        "models",
        "electra_base",
        "checkpoint-4000",
    ),
)

TOKENIZER_NAME = os.environ.get(
    "TOKENIZER_NAME",
    "google/electra-base-discriminator",
)

THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
UNIT_TEST_MODE = os.environ.get("UNIT_TEST_MODE", "0") == "1"

predictor = None
if not UNIT_TEST_MODE:
    from src.api.predictor import PredictorConfig, SentimentPredictor

    predictor = SentimentPredictor(
        PredictorConfig(
            model_path=MODEL_PATH,
            tokenizer_name=TOKENIZER_NAME,
            threshold=THRESHOLD,
        )
    )

HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Air Paradis - Sentiment</title>
  </head>
  <body style="font-family: Arial; max-width: 800px; margin: 40px auto;">
    <h2>Air Paradis - Prédiction de sentiment</h2>
    <form method="post" action="/">
      <textarea name="text" rows="4" style="width: 100%;" placeholder="Écris un tweet...">{{ text }}</textarea>
      <br/><br/>
      <button type="submit">Prédire</button>
    </form>

    {% if result %}
      <hr/>
      <h3>Résultat</h3>
      <pre>{{ result }}</pre>
    {% endif %}
  </body>
</html>
"""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": "electra_transformer" if not UNIT_TEST_MODE else "mock_predictor",
        "unit_test_mode": UNIT_TEST_MODE,
        "model_path": MODEL_PATH,
        "tokenizer_name": TOKENIZER_NAME,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    result = None

    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip():
            if UNIT_TEST_MODE:
                result = {
                    "tweet": text,
                    "tweet_clean": text.strip(),
                    "proba_pos": 0.9,
                    "proba_neg": 0.1,
                    "pred_label": 1,
                    "pred_text": "positif",
                    "threshold": THRESHOLD,
                    "bad_buzz": False,
                }
            else:
                result = predictor.predict_one(text)
        else:
            result = {"error": "Texte vide."}

    return render_template_string(HTML, text=text, result=result)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing field 'text' (non-empty string)."}), 400

    if UNIT_TEST_MODE:
        return jsonify({
            "tweet": text,
            "tweet_clean": text.strip(),
            "proba_pos": 0.9,
            "proba_neg": 0.1,
            "pred_label": 1,
            "pred_text": "positif",
            "threshold": THRESHOLD,
            "bad_buzz": False,
        }), 200

    out = predictor.predict_one(text)
    return jsonify(out), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)