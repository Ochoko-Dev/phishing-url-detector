import os
import json
import time
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.extract_features import FeatureExtractor

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

app = FastAPI(title="Phishing URL Detection API")

# Loaded once at startup
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)


class URLRequest(BaseModel):
    url: str


@app.get("/")
def health_check():
    return {"status": "ok", "models_loaded": ["random_forest", "logistic_regression"]}


@app.post("/predict")
def predict(request: URLRequest):
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL must not be empty.")

    start_time = time.time()

    raw_features = FeatureExtractor.extract_features(url)
    # Only the exact feature set and order the models were trained on.
    # path_length and count_slash are deliberately excluded (see train.py) --
    # they were found to be a dataset-construction artifact, not a genuine
    # phishing signal.
    row = pd.DataFrame([{col: raw_features[col] for col in FEATURE_COLS}])

    rf_proba = rf_model.predict_proba(row)[0][1]
    rf_is_phishing = bool(rf_proba >= 0.5)

    row_scaled = scaler.transform(row)
    lr_proba = lr_model.predict_proba(row_scaled)[0][1]
    lr_is_phishing = bool(lr_proba >= 0.5)

    latency_ms = (time.time() - start_time) * 1000

    return {
        "url": url,
        "latency_ms": round(latency_ms, 4),
        "models": {
            "random_forest": {
                "is_phishing": rf_is_phishing,
                "confidence_score": round(float(rf_proba if rf_is_phishing else 1 - rf_proba), 4),
            },
            "logistic_regression": {
                "is_phishing": lr_is_phishing,
                "confidence_score": round(float(lr_proba if lr_is_phishing else 1 - lr_proba), 4),
            },
        },
        "agreement": rf_is_phishing == lr_is_phishing,
        "raw_features": raw_features,
    }
