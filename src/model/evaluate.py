import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "data/processed/dataset.csv"
MODEL_DIR = "models"

def main():
    print("[*] Loading dataset and models for deep evaluation...")
    df = pd.read_csv(DATASET_PATH)
    
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    feature_cols = [col for col in df.columns if col not in ['url', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Load Models
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
    
    print("\n" + "="*60)
    print(" RANDOM FOREST EVALUATION REPORT")
    print("="*60)
    rf_preds = rf_model.predict(X_test)
    print(classification_report(y_test, rf_preds, target_names=['Safe (0)', 'Phishing (1)'], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))
    
    print("\n" + "="*60)
    print(" XGBOOST EVALUATION REPORT")
    print("="*60)
    xgb_preds = xgb_model.predict(X_test)
    print(classification_report(y_test, xgb_preds, target_names=['Safe (0)', 'Phishing (1)'], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, xgb_preds))
    
    print("\n" + "="*60)
    print(" TOP 5 FEATURE IMPORTANCES (XGBoost)")
    print("="*60)
    importances = xgb_model.feature_importances_
    feature_ranking = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    for feat, score in feature_ranking.head(5).items():
        print(f"  - {feat:<25}: {score:.4f}")

if __name__ == "__main__":
    main()
