import os
import sys
import pickle
import scipy.sparse as sp
from flask import Flask, request, jsonify, render_template
from werkzeug.middleware.proxy_fix import ProxyFix

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "model-2.pkl")         # <-- your file
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "vectorizer.pkl")

LABELS = {0: "ham", 1: "spam"}

# ---------- Helpers ----------
def _fail(msg: str):
    print(f"[BOOT ERROR] {msg}", file=sys.stderr)
    raise SystemExit(1)

def _to_dense(X):
    """Ensure dense for models trained on dense arrays (e.g., SVC without sparse support)."""
    return X.toarray() if sp.issparse(X) else X

# ---------- Load artifacts ----------
if not os.path.exists(VECTORIZER_PATH):
    _fail(f"Missing vectorizer file: {VECTORIZER_PATH}")

if not os.path.exists(MODEL_PATH):
    _fail(f"Missing model file: {MODEL_PATH} (set MODEL_PATH env var or place the file here)")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Loaded model type:", type(model))
print("Loaded vectorizer type:", type(vectorizer))

# ---------- App ----------
app = Flask(__name__)
# Make Flask play nice behind proxies (Heroku/Render)
app.wsgi_app = ProxyFix(app.wsgi_app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_form():
    text = request.form.get("message", "")
    if not isinstance(text, str) or not text.strip():
        return render_template("index.html", error="Please enter a message."), 400

    X = _to_dense(vectorizer.transform([text]))
    pred = model.predict(X)[0]

    label = LABELS.get(int(pred), str(pred))
    proba = None
    if hasattr(model, "predict_proba"):
        # index 1 == probability of spam if your model uses {0: ham, 1: spam}
        proba = float(model.predict_proba(X)[0][1])

    return render_template("index.html", result=label, proba=proba, message=text)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    text = data.get("message", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Provide non-empty 'message' (string)."}), 400

    X = _to_dense(vectorizer.transform([text]))
    pred = model.predict(X)[0]

    resp = {"label": LABELS.get(int(pred), str(pred))}
    if hasattr(model, "predict_proba"):
        resp["spam_probability"] = float(model.predict_proba(X)[0][1])

    return jsonify(resp), 200

if __name__ == "__main__":
    # Local dev server (in production use gunicorn via Procfile)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
