import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "data/processed/dataset.csv"
KENYA_RAW_PATH = "data/raw/kenya_manual.csv"
MODEL_DIR = "models"

def main():
    print("[*] Loading dataset and models for deep evaluation...")
    df = pd.read_csv(DATASET_PATH)

    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    kenya_urls = set()
    if os.path.exists(KENYA_RAW_PATH):
        df_kenya_raw = pd.read_csv(KENYA_RAW_PATH)
        kenya_urls = set(df_kenya_raw['url'].astype(str).str.strip())

    is_kenya = df['url'].astype(str).str.strip().isin(kenya_urls)
    df_kenya = df[is_kenya].reset_index(drop=True)
    df_main = df[~is_kenya].reset_index(drop=True)

    X = df_main[feature_cols]
    y = df_main['label']
    X_kenya = df_kenya[feature_cols]
    y_kenya = df_kenya['label']

    # Same split as train.py -- sample weights don't matter for re-deriving
    # the test split itself, only random_state and stratify do.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Load Models
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    print("\n" + "=" * 60)
    print(" RANDOM FOREST EVALUATION REPORT (general test set)")
    print("=" * 60)
    rf_preds = rf_model.predict(X_test)
    print(classification_report(y_test, rf_preds, target_names=['Safe (0)', 'Phishing (1)'], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))

    if len(y_kenya) > 0:
        print("\n" + "-" * 60)
        print(" RANDOM FOREST EVALUATION REPORT (Kenya-context holdout)")
        print("-" * 60)
        rf_kenya_preds = rf_model.predict(X_kenya)
        print(classification_report(y_kenya, rf_kenya_preds, target_names=['Safe (0)', 'Phishing (1)'], digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_kenya, rf_kenya_preds))

    print("\n" + "=" * 60)
    print(" LOGISTIC REGRESSION EVALUATION REPORT (general test set)")
    print("=" * 60)
    X_test_scaled = scaler.transform(X_test)
    lr_preds = lr_model.predict(X_test_scaled)
    print(classification_report(y_test, lr_preds, target_names=['Safe (0)', 'Phishing (1)'], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_preds))

    if len(y_kenya) > 0:
        print("\n" + "-" * 60)
        print(" LOGISTIC REGRESSION EVALUATION REPORT (Kenya-context holdout)")
        print("-" * 60)
        X_kenya_scaled = scaler.transform(X_kenya)
        lr_kenya_preds = lr_model.predict(X_kenya_scaled)
        print(classification_report(y_kenya, lr_kenya_preds, target_names=['Safe (0)', 'Phishing (1)'], digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_kenya, lr_kenya_preds))

    print("\n" + "=" * 60)
    print(" TOP 5 FEATURE IMPORTANCES (Random Forest)")
    print("=" * 60)
    importances = rf_model.feature_importances_
    feature_ranking = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    for feat, score in feature_ranking.head(5).items():
        print(f"  - {feat:<25}: {score:.4f}")

if __name__ == "__main__":
    main()
