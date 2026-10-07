import os
import sys
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# Add root directory to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.extract_features import FeatureExtractor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RF_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
XGB_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.json")

def load_models():
    models = {}
    if os.path.exists(XGB_PATH):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(XGB_PATH)
        models['xgboost'] = xgb_model
    if os.path.exists(RF_PATH):
        models['random_forest'] = joblib.load(RF_PATH)
    return models

def predict_single_url(url: str, model_type: str = "xgboost"):
    models = load_models()
    if not models:
        print("[!] Error: No trained models found in 'models/' directory.")
        return

    selected_model = models.get(model_type, list(models.values())[0])
    model_name = "XGBoost" if model_type == "xgboost" and "xgboost" in models else "Random Forest"

    start_time = time.time()
    
    # Extract features dynamically using FeatureExtractor
    features = FeatureExtractor.extract(url)
    input_df = pd.DataFrame([features])
    
    # Predict
    prob = float(selected_model.predict_proba(input_df)[0][1])
    is_phishing = prob >= 0.5
    confidence = prob if is_phishing else 1.0 - prob
    latency = (time.time() - start_time) * 1000

    print("\n" + "=" * 55)
    print(" PHISHING URL DETECTION RESULT")
    print("=" * 55)
    print(f" Target URL   : {url}")
    print(f" Result       : {'[!] PHISHING DETECTED' if is_phishing else '[+] SAFE URL'}")
    print(f" Confidence   : {confidence * 100:.2f}%")
    print(f" Model Used   : {model_name}")
    print(f" Latency      : {latency:.2f} ms")
    print("=" * 55 + "\n")

def main():
    parser = argparse.ArgumentParser(description="CLI Tool for Real-Time Phishing URL Detection")
    parser.add_argument("-u", "--url", type=str, help="Single URL to analyze")
    parser.add_argument("-m", "--model", type=str, choices=["xgboost", "random_forest"], default="xgboost", help="ML model to use (default: xgboost)")
    
    args = parser.parse_args()

    if args.url:
        predict_single_url(args.url, args.model)
    else:
        print("=== Interactive Phishing Detector CLI (Type 'exit' or 'q' to quit) ===")
        while True:
            target = input("\nEnter URL to inspect: ").strip()
            if target.lower() in ['exit', 'q']:
                print("Exiting scanner...")
                break
            if target:
                predict_single_url(target, args.model)

if __name__ == "__main__":
    main()
