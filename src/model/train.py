import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# File Paths
DATASET_PATH = "data/processed/dataset.csv"
MODEL_DIR = "models"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("[*] Loading processed dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}. Run build_dataset.py first.")
        
    df = pd.read_csv(DATASET_PATH)
    
    # Ensure label column is numeric
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    # Filter out non-feature columns
    feature_cols = [col for col in df.columns if col not in ['url', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    print(f"[+] Loaded {X.shape[0]} samples with {X.shape[1]} numerical features.")
    
    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"[+] Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    print("-" * 60)
    
    # ------------------------------------------------------------------
    # 1. Train Random Forest Classifier
    # ------------------------------------------------------------------
    print("[*] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - start_time
    
    # Latency & Prediction Test
    start_time = time.time()
    rf_preds = rf_model.predict(X_test)
    rf_latency = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    
    print(f"[+] RF Trained in {rf_train_time:.2f}s | Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f} | Latency: {rf_latency:.4f} ms/sample")
    
    # ------------------------------------------------------------------
    # 2. Train XGBoost Classifier
    # ------------------------------------------------------------------
    print("[*] Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    xgb_train_time = time.time() - start_time
    
    # Latency & Prediction Test
    start_time = time.time()
    xgb_preds = xgb_model.predict(X_test)
    xgb_latency = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_prec = precision_score(y_test, xgb_preds)
    xgb_rec = recall_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds)
    
    print(f"[+] XGB Trained in {xgb_train_time:.2f}s | Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | Latency: {xgb_latency:.4f} ms/sample")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 3. Save Trained Models
    # ------------------------------------------------------------------
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    
    joblib.dump(rf_model, rf_path)
    xgb_model.save_model(xgb_path)
    
    print(f"[SUCCESS] Models saved cleanly:")
    print(f"  - Random Forest -> {rf_path}")
    print(f"  - XGBoost -> {xgb_path}")

if __name__ == "__main__":
    main()
