import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model
model = joblib.load("../models/logistic_model.pkl")

# Load your test CSV
df = pd.read_csv("../data/telco_clean.csv")

# Features used in the model
FEATURES = [
    "Gender",
    "Tenure Months",
    "Internet Service",
    "Streaming Movies",
    "Monthly Charges",
    "Total Charges"
]

X_test = df[FEATURES]

# ⚠ Churn is already 0 and 1 → just convert to int
y_test = df["Churn"].astype(int)

# Predict
preds = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)
matrix = confusion_matrix(y_test, preds)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", matrix)
