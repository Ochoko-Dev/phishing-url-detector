import os
import pandas as pd
from extract_features import FeatureExtractor

# File Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "dataset.csv")

def load_and_standardize_datasets():
    datasets = []

    # 1. Load Local Manual Dataset (Kenya specific)
    kenya_path = os.path.join(DATA_RAW_DIR, "kenya_manual.csv")
    if os.path.exists(kenya_path):
        df_kenya = pd.read_csv(kenya_path)
        if 'url' in df_kenya.columns and 'label' in df_kenya.columns:
            df_kenya = df_kenya[['url', 'label']]
            datasets.append(df_kenya)
            print(f"[+] Loaded {len(df_kenya)} rows from local manual dataset.")

    # 2. Load URLhaus (Phishing Base - Label 1)
    urlhaus_path = os.path.join(DATA_RAW_DIR, "urlhaus.csv")
    phish_count = 0
    if os.path.exists(urlhaus_path):
        try:
            df_urlhaus = pd.read_csv(urlhaus_path, skiprows=8, header=None)
            url_col = None
            for col in df_urlhaus.columns:
                if df_urlhaus[col].astype(str).str.contains('http').any():
                    url_col = col
                    break
            if url_col is not None:
                df_phish = df_urlhaus[[url_col]].rename(columns={url_col: 'url'})
                df_phish['label'] = 1
                datasets.append(df_phish)
                phish_count = len(df_phish)
                print(f"[+] Loaded {phish_count} rows from URLhaus (Label 1).")
        except Exception as e:
            print(f"[-] Warning: Failed to load URLhaus: {e}")

    # 3. Load Tranco Domains (Legitimate Base - Label 0)
    tranco_path = os.path.join(DATA_RAW_DIR, "tranco.csv")
    if os.path.exists(tranco_path):
        try:
            # Tranco comes as rank,domain without headers
            df_tranco = pd.read_csv(tranco_path, header=None, names=['rank', 'url'])
            # Sample Tranco to match URLhaus count for a 1:1 balanced ratio
            sample_size = phish_count if phish_count > 0 else 15000
            df_tranco = df_tranco[['url']].head(sample_size)
            df_tranco['label'] = 0
            datasets.append(df_tranco)
            print(f"[+] Loaded {len(df_tranco)} rows from Tranco (Label 0).")
        except Exception as e:
            print(f"[-] Warning: Failed to load Tranco: {e}")

    if not datasets:
        raise ValueError("No raw dataset files found in data/raw/")

    # Combine and clean
    df_combined = pd.concat(datasets, ignore_index=True)
    df_combined['url'] = df_combined['url'].astype(str).str.strip()
    df_combined = df_combined.dropna(subset=['url'])
    df_combined = df_combined.drop_duplicates(subset=['url'])
    
    print(f"[+] Total unique URLs merged: {len(df_combined)}")
    return df_combined

def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    print("[*] Processing raw datasets...")
    
    df = load_and_standardize_datasets()
    
    print("[*] Extracting features from URLs (this will take ~15 seconds)...")
    
    features_list = [FeatureExtractor.extract(url) for url in df['url']]
    df_features = pd.DataFrame(features_list)
    
    df_final = pd.concat([df.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    # Class Distribution Output
    counts = df_final['label'].value_counts().to_dict()
    print(f"[SUCCESS] Dataset built successfully with {df_final.shape[0]} rows and {df_final.shape[1]} columns.")
    print(f"[+] Class Balance -> Label 0 (Safe): {counts.get(0, 0)} | Label 1 (Phishing): {counts.get(1, 0)}")
    print(f"[+] Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
