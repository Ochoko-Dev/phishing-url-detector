import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# File Paths
DATASET_PATH = "data/processed/dataset.csv"
KENYA_RAW_PATH = "data/raw/kenya_manual.csv"
MODEL_DIR = "models"

# Features dropped from training. path_length and count_slash were found to be
# dataset-construction artifacts, not genuine phishing signals: 99.99% of the
# legitimate class (PhiUSIIL) are bare root-domain URLs (path_length=0), while
# phishing feeds report full attack URLs with real paths. The classifier was
# learning "has a path = phishing", causing a 100% false-positive rate on real
# Kenyan bank URLs in the holdout evaluation. Dropping both features recovered
# most of that loss with a negligible cost to general test performance
# (Random Forest: -0.03pp accuracy; Logistic Regression: improved instead).
DROPPED_FEATURES = ['path_length', 'count_slash']

# Sample weight applied to rows where brand_present==1 (the domain contains a
# known Kenyan bank/mobile-money brand name). This subset is where the
# brand_domain_mismatch feature -- a well-targeted impersonation signal -- is
# meaningful, but it covers only ~0.08% of the merged dataset (248 of 326,088
# rows), so the model otherwise never learns to rely on it. Weighting forces
# it into consideration during training. See Discussion chapter.
BRAND_PRESENT_SAMPLE_WEIGHT = 100.0

def evaluate(model, name, X_test, y_test, X_kenya, y_kenya):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"[+] {name} (general test set, n={len(y_test)}) -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    if len(y_kenya) > 0:
        kenya_preds = model.predict(X_kenya)
        k_acc = accuracy_score(y_kenya, kenya_preds)
        k_prec = precision_score(y_kenya, kenya_preds, zero_division=0)
        k_rec = recall_score(y_kenya, kenya_preds, zero_division=0)
        k_f1 = f1_score(y_kenya, kenya_preds, zero_division=0)
        print(f"[+] {name} (Kenya-context holdout, n={len(y_kenya)}) -> Accuracy: {k_acc:.4f} | Precision: {k_prec:.4f} | Recall: {k_rec:.4f} | F1: {k_f1:.4f}")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[*] Loading processed dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}. Run build_dataset.py first.")

    df = pd.read_csv(DATASET_PATH)

    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    feature_cols = [col for col in df.columns if col not in ['url', 'label'] + DROPPED_FEATURES]

    # --- Carve out the Kenya-context manual set as a dedicated held-out test split ---
    kenya_urls = set()
    if os.path.exists(KENYA_RAW_PATH):
        df_kenya_raw = pd.read_csv(KENYA_RAW_PATH)
        kenya_urls = set(df_kenya_raw['url'].astype(str).str.strip())

    is_kenya = df['url'].astype(str).str.strip().isin(kenya_urls)
    df_kenya_holdout = df[is_kenya].reset_index(drop=True)
    df_main = df[~is_kenya].reset_index(drop=True)

    print(f"[+] Held out {df_kenya_holdout.shape[0]} Kenya-context rows as a dedicated test set (never trained on).")
    print(f"[+] Remaining pool for train/test split: {df_main.shape[0]} rows.")
    print(f"[+] Dropped features (dataset-artifact mitigation): {DROPPED_FEATURES}")

    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"[+] Saved feature column order to {os.path.join(MODEL_DIR, 'feature_cols.json')}")

    sample_weights_all = np.where(df_main['brand_present'] == 1, BRAND_PRESENT_SAMPLE_WEIGHT, 1.0)
    print(f"[+] {int((df_main['brand_present'] == 1).sum())} rows in the training pool flagged for {BRAND_PRESENT_SAMPLE_WEIGHT:.0f}x sample weight (brand_present==1).")

    X = df_main[feature_cols]
    y = df_main['label']

    X_kenya = df_kenya_holdout[feature_cols]
    y_kenya = df_kenya_holdout['label']

    print(f"[+] Loaded {X.shape[0]} samples with {X.shape[1]} numerical features.")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights_all, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[+] Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    print("-" * 60)

    # ------------------------------------------------------------
    # 1. Train Random Forest Classifier
    # ------------------------------------------------------------
    print("[*] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    start_time = time.time()
    rf_model.fit(X_train, y_train, sample_weight=w_train)
    rf_train_time = time.time() - start_time

    start_time = time.time()
    rf_preds = rf_model.predict(X_test)
    rf_latency = (time.time() - start_time) / len(X_test) * 1000

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)

    print(f"[+] RF Trained in {rf_train_time:.2f}s | Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f} | Latency: {rf_latency:.4f} ms/sample")
    evaluate(rf_model, "Random Forest", X_test, y_test, X_kenya, y_kenya)

    print("-" * 60)

    # ------------------------------------------------------------
    # 2. Train Logistic Regression Classifier
    # ------------------------------------------------------------
    print("[*] Training Logistic Regression Classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_kenya_scaled = scaler.transform(X_kenya) if len(X_kenya) > 0 else X_kenya

    lr_model = LogisticRegression(max_iter=1000, random_state=42)

    start_time = time.time()
    lr_model.fit(X_train_scaled, y_train, sample_weight=w_train)
    lr_train_time = time.time() - start_time

    start_time = time.time()
    lr_preds = lr_model.predict(X_test_scaled)
    lr_latency = (time.time() - start_time) / len(X_test) * 1000

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)

    print(f"[+] LR Trained in {lr_train_time:.2f}s | Accuracy: {lr_acc:.4f} | F1: {lr_f1:.4f} | Latency: {lr_latency:.4f} ms/sample")
    evaluate(lr_model, "Logistic Regression", X_test_scaled, y_test, X_kenya_scaled, y_kenya)

    print("=" * 60)

    # ------------------------------------------------------------
    # 3. Save Trained Models
    # ------------------------------------------------------------
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    lr_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    joblib.dump(rf_model, rf_path)
    joblib.dump(lr_model, lr_path)
    joblib.dump(scaler, scaler_path)

    print(f"[SUCCESS] Models saved cleanly:")
    print(f"  - Random Forest -> {rf_path}")
    print(f"  - Logistic Regression -> {lr_path}")
    print(f"  - Feature scaler (required for LR inference) -> {scaler_path}")
    print(f"  - Feature column order -> {os.path.join(MODEL_DIR, 'feature_cols.json')}")

if __name__ == "__main__":
    main()
