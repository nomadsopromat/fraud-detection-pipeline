from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime, timezone

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/fraud_model.pkl")
amount_scaler = joblib.load("models/amount_scaler.pkl")
time_scaler = joblib.load("models/time_scaler.pkl")

THRESHOLD = 0.3
LOG_FILE = "logs/predictions.csv"

os.makedirs("logs", exist_ok=True)

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

def log_prediction(data: dict, amount_scaled: float, time_scaled: float, proba: float, is_fraud: bool):
    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **data,
        "Amount_scaled": amount_scaled,
        "Time_scaled": time_scaled,
        "fraud_probability": proba,
        "is_fraud": is_fraud,
    }
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

@app.get("/")
def root():
    return {"status": "Fraud detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.model_dump()

    amount_scaled = amount_scaler.transform([[data["Amount"]]])[0][0]
    time_scaled = time_scaler.transform([[data["Time"]]])[0][0]

    feature_row = {f"V{i}": data[f"V{i}"] for i in range(1, 29)}
    feature_row["Amount_scaled"] = amount_scaled
    feature_row["Time_scaled"] = time_scaled

    X = pd.DataFrame([feature_row])

    proba = model.predict_proba(X)[0][1]
    is_fraud = bool(proba >= THRESHOLD)

    log_prediction(data, amount_scaled, time_scaled, float(proba), is_fraud)

    return {
        "fraud_probability": float(proba),
        "is_fraud": is_fraud,
        "threshold_used": THRESHOLD
    }