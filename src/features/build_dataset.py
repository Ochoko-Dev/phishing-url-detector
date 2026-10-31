import os
import pandas as pd
from extract_features import FeatureExtractor

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "dataset.csv")

def load_and_standardize_datasets():
    datasets = []

    # 1. Kenya manual (both classes) - highest-value, Kenya-context data
    kenya_path = os.path.join(DATA_RAW_DIR, "kenya_manual.csv")
    if os.path.exists(kenya_path):
        df_kenya = pd.read_csv(kenya_path)
        if 'url' in df_kenya.columns and 'label' in df_kenya.columns:
            df_kenya = df_kenya[['url', 'label']]
            datasets.append(df_kenya)
            print(f"[+] Loaded {len(df_kenya)} rows from kenya_manual.csv.")

    # 2. PhiUSIIL (both classes).
    # NOTE: PhiUSIIL's native convention is label=1 -> LEGITIMATE, label=0 -> PHISHING,
    # the OPPOSITE of this project's convention (label=1 -> phishing). Flipped on load.
    phiusiil_path = os.path.join(DATA_RAW_DIR, "phiusiil.csv")
    if os.path.exists(phiusiil_path):
        df_phiusiil = pd.read_csv(phiusiil_path, usecols=['URL', 'label'])
        df_phiusiil = df_phiusiil.rename(columns={'URL': 'url'})
        df_phiusiil['label'] = 1 - df_phiusiil['label']
        datasets.append(df_phiusiil)
        print(f"[+] Loaded {len(df_phiusiil)} rows from PhiUSIIL (label convention flipped).")

    # 3. PhishTank (phishing only, Label 1) - keep only verified+online reports
    phishtank_path = os.path.join(DATA_RAW_DIR, "phishtank.csv")
    if os.path.exists(phishtank_path):
        df_phishtank = pd.read_csv(phishtank_path)
        df_phishtank = df_phishtank[(df_phishtank['verified'] == 'yes') & (df_phishtank['online'] == 'yes')]
        df_phishtank = df_phishtank[['url']].copy()
        df_phishtank['label'] = 1
        datasets.append(df_phishtank)
        print(f"[+] Loaded {len(df_phishtank)} verified/online rows from PhishTank (Label 1).")

    # 4. OpenPhish (phishing only, Label 1) - plain text, one URL per line
    openphish_path = os.path.join(DATA_RAW_DIR, "openphish.txt")
    if os.path.exists(openphish_path):
        with open(openphish_path) as f:
            urls = [line.strip() for line in f if line.strip()]
        df_openphish = pd.DataFrame({'url': urls})
        df_openphish['label'] = 1
        datasets.append(df_openphish)
        print(f"[+] Loaded {len(df_openphish)} rows from OpenPhish (Label 1).")

    # 5. URLhaus (phishing/malware base, Label 1)
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
    # 6. Legit-paths supplement (both hand-verified, both label=0, with real non-root paths)
    # Added to counter a dataset-construction bias: ~99.99% of PhiUSIIL's
    # legitimate-class rows are bare root-domain URLs (path_length=0), while
    # the phishing class is dominated by full attack URLs with real paths.
    # The classifier was found to learn "has a path = phishing" as a spurious
    # shortcut. This supplement alone is far too small to fix that numerically
    # (17 rows vs 134,846); it is combined with sample_weight boosting in
    # train.py. See data/SOURCES.md and the Discussion chapter for the finding.
    legit_paths_path = os.path.join(DATA_RAW_DIR, "legit_paths_supplement.csv")
    if os.path.exists(legit_paths_path):
        df_legit_paths = pd.read_csv(legit_paths_path)
        if 'url' in df_legit_paths.columns and 'label' in df_legit_paths.columns:
            df_legit_paths = df_legit_paths[['url', 'label']]
            datasets.append(df_legit_paths)
            print(f"[+] Loaded {len(df_legit_paths)} rows from legit_paths_supplement.csv.")
    # Tranco (top-ranked bare domains) was evaluated and deliberately EXCLUDED:
    # (1) entries are bare domains with no path, while every phishing source here
    #     is a full URL with a path/subdomain -- mixing them would let the model
    #     learn "has a path = phishing" instead of real lexical signals.
    # (2) Tranco's top-N ranking skews toward globally famous mega-brands only,
    #     not representative "legitimate but unknown" sites.
    # PhiUSIIL's legitimate class (full URLs, broad domain variety) is used
    # instead. See data/SOURCES.md for the documented decision.

    if not datasets:
        raise ValueError("No raw dataset files found in data/raw/")

    df_combined = pd.concat(datasets, ignore_index=True)
    df_combined['url'] = df_combined['url'].astype(str).str.strip()
    df_combined = df_combined.dropna(subset=['url'])
    df_combined = df_combined.drop_duplicates(subset=['url'])

    print(f"[+] Total unique URLs merged: {len(df_combined)}")
    counts = df_combined['label'].value_counts().to_dict()
    print(f"[+] Class split -> Label 0 (Safe): {counts.get(0, 0)} | Label 1 (Phishing): {counts.get(1, 0)}")
    return df_combined

def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    print("[*] Processing raw datasets...")

    df = load_and_standardize_datasets()

    print("[*] Extracting features from URLs (this may take a while for this volume)...")

    features_list = [FeatureExtractor.extract_features(url) for url in df['url']]
    df_features = pd.DataFrame(features_list)

    df_final = pd.concat([df.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)

    df_final.to_csv(OUTPUT_FILE, index=False)

    counts = df_final['label'].value_counts().to_dict()
    print(f"[SUCCESS] Dataset built successfully with {df_final.shape[0]} rows and {df_final.shape[1]} columns.")
    print(f"[+] Class Balance -> Label 0 (Safe): {counts.get(0, 0)} | Label 1 (Phishing): {counts.get(1, 0)}")
    print(f"[+] Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
