# app.py
"""Churn Insight API — Flask backend (clean + production-ready for Render)."""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from dotenv import load_dotenv

# Load .env locally (Render uses environment variables set in dashboard)
load_dotenv()

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# Load model once at startup
try:
    log_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

@app.get("/")
def root():
    return jsonify({"status": "ok", "message": "Churn Insight API Running"}), 200


# ✅ CLEAN + FIXED PREDICT ENDPOINT (NO MONGO)
@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # Required input fields
    required_fields = [
        "Gender", "Tenure Months", "Internet Service",
        "Streaming Movies", "Monthly Charges", "Total Charges"
    ]

    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "fields": missing}), 400

    # Convert request into a DataFrame
    try:
        df = pd.DataFrame([{
            "Gender": data["Gender"],
            "Tenure Months": float(data["Tenure Months"]),
            "Internet Service": data["Internet Service"],
            "Streaming Movies": data["Streaming Movies"],
            "Monthly Charges": float(data["Monthly Charges"]),
            "Total Charges": float(data["Total Charges"])
        }])
    except Exception as exc:
        return jsonify({"error": "Invalid field types", "message": str(exc)}), 400

    # Make prediction
    pred = int(log_model.predict(df)[0])
    prob = float(log_model.predict_proba(df)[0][1])

    result = {
        "customer_input": data,
        "logistic_regression": {
            "prediction": "Yes" if pred == 1 else "No",
            "probability": round(prob, 4),
            "risk_percent": round(prob * 100, 2)
        }
    }

    return jsonify(result), 200


# ✅ GET MODEL ACCURACY
@app.get("/accuracy")
def get_accuracy():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "metrics.json missing"}), 404
    
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    return jsonify(metrics), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
