import os
import time
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# Import your existing feature extraction module
from src.features.extract_features import FeatureExtractor

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RF_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
XGB_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.json")

# Initialize FastAPI App
app = FastAPI(
    title="Phishing URL Detection API",
    description="Real-time lexical and structural feature classification endpoint.",
    version="1.0.0"
)

# Global variables for models
rf_model = None
xgb_model = None

@app.on_event("startup")
def load_models():
    global rf_model, xgb_model
    print("[*] Loading trained machine learning models...")
    
    if os.path.exists(RF_PATH):
        rf_model = joblib.load(RF_PATH)
        print("    [+] Loaded Random Forest model.")
    else:
        print(f"    [-] Warning: Random Forest model not found at {RF_PATH}")
        
    if os.path.exists(XGB_PATH):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(XGB_PATH)
        print("    [+] Loaded XGBoost model.")
    else:
        print(f"    [-] Warning: XGBoost model not found at {XGB_PATH}")

class URLRequest(BaseModel):
    url: str

class DetectionResponse(BaseModel):
    url: str
    prediction: str
    is_phishing: bool
    confidence_score: float
    latency_ms: float
    model_used: str

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Phishing URL Detection Engine",
        "models_loaded": {
            "random_forest": rf_model is not None,
            "xgboost": xgb_model is not None
        }
    }

@app.post("/predict", response_model=DetectionResponse)
def predict_url(payload: URLRequest):
    if not xgb_model and not rf_model:
        raise HTTPException(status_code=500, detail="Machine learning models are not loaded.")

    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    start_time = time.time()

    # 1. Dynamically extract features from raw URL input
    features_dict = FeatureExtractor.extract(url)
    
    # 2. Format features as Pandas DataFrame matching model input format
    input_df = pd.DataFrame([features_dict])

    # 3. Perform Inference using preferred model (XGBoost)
    if xgb_model:
        prob = float(xgb_model.predict_proba(input_df)[0][1])
        model_name = "XGBoost"
    else:
        prob = float(rf_model.predict_proba(input_df)[0][1])
        model_name = "Random Forest"

    is_phishing = prob >= 0.5
    prediction = "Phishing" if is_phishing else "Safe"
    latency = (time.time() - start_time) * 1000  # convert to ms

    return {
        "url": url,
        "prediction": prediction,
        "is_phishing": is_phishing,
        "confidence_score": round(prob if is_phishing else 1.0 - prob, 4),
        "latency_ms": round(latency, 2),
        "model_used": model_name
    }
