# server/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from pymongo import MongoClient
import json

# ================================
#  MONGODB CONNECTION
# ================================
MONGO_URI = "mongodb+srv://muzammilalambca23_db_user:alam9008@cluster0.alzkc0w.mongodb.net/"
client = MongoClient(MONGO_URI)

db = client["churn_db"]
predictions_collection = db["predictions"]

# ================================
#  FLASK APP CONFIG
# ================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ================================
#  FILE PATHS (FIXED)
# ================================
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models/logistic_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models/metrics.json")  # ✅ FIXED


# Load Model
log_model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return jsonify({"status": "ok", "message": "Churn Insight API Running"})


# ================================
#  PREDICT API (Final Clean Version)
# ================================
@app.post("/predict")
def predict():
    data = request.get_json()

    required_fields = [
        "Gender", "Tenure Months", "Internet Service",
        "Streaming Movies", "Monthly Charges", "Total Charges"
    ]
    for f in required_fields:
        if f not in data:
            return jsonify({"error": f"Missing field: {f}"}), 400

    # Build DataFrame
    try:
        df = pd.DataFrame([{
            "Gender": data["Gender"],
            "Tenure Months": float(data["Tenure Months"]),
            "Internet Service": data["Internet Service"],
            "Streaming Movies": data["Streaming Movies"],
            "Monthly Charges": float(data["Monthly Charges"]),
            "Total Charges": float(data["Total Charges"])
        }])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Predict
    pred = log_model.predict(df)[0]
    prob = log_model.predict_proba(df)[0][1]

    result = {
        "customer_input": data,
        "logistic_regression": {
            "prediction": "Yes" if pred == 1 else "No",
            "probability": round(float(prob), 4),
            "risk_percent": round(float(prob) * 100, 2)
        }
    }

    # Save a CLEAN copy to MongoDB
    mongo_copy = json.loads(json.dumps(result))
    predictions_collection.insert_one(mongo_copy)

    return jsonify(result), 201


# ================================
#  GET PREDICTIONS (CLEAN)
# ================================
@app.get("/predictions")
def get_predictions():
    docs = list(predictions_collection.find({}, {"_id": 0}))  # remove _id
    docs.reverse()  # newest first
    return jsonify(docs)


# ================================
#  GET ACCURACY (Metrics)
# ================================
@app.get("/accuracy")
def get_accuracy():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "metrics.json missing"}), 404

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    return jsonify(metrics)


# ================================
#  RUN SERVER (LAN Friendly)
# ================================
# ================================
#  RUN SERVER (LAN Friendly)
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

