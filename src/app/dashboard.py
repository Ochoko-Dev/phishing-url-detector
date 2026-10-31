import streamlit as st
import requests
import pandas as pd
import sys
import os

# Add root directory to sys.path to access feature extractor directly if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.extract_features import FeatureExtractor

# Page Configuration
st.set_page_config(
    page_title="Phishing URL Detection Engine",
    page_icon="🛡️",
    layout="wide"
)

import os as _os
BACKEND_BASE_URL = _os.environ.get("BACKEND_URL", st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")) if hasattr(st, "secrets") else _os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = f"{BACKEND_BASE_URL}/predict"
st.title("🛡️ Phishing URL Threat Detection Engine")
st.markdown("Real-time lexical and structural URL analysis powered by Machine Learning.")

# Sidebar Controls
st.sidebar.header("Configuration & API Status")
try:
    health_check = requests.get(BACKEND_BASE_URL + "/", timeout=2)
    if health_check.status_code == 200:
        st.sidebar.success("FastAPI Backend: ONLINE")
    else:
        st.sidebar.warning("FastAPI Backend: UNHEALTHY")
except Exception:
    st.sidebar.error("FastAPI Backend: OFFLINE")

st.sidebar.info(
    "Framework Architecture:\n"
    "- Backend: FastAPI\n"
    "- ML Models: Random Forest / Logistic Regression\n"
    "- Latency Target: < 2.0s\n\n"
    "Both models are served and shown side by side -- they were found to "
    "generalize differently on Kenyan banking URLs during evaluation "
    "(see Discussion chapter), so a single 'primary' model would hide "
    "that tradeoff."
)

# Main Interface
target_url = st.text_input("Enter URL to scan:", placeholder="e.g. http://equity-online-verify-ke.com/login")

if st.button("Scan URL", type="primary"):
    if not target_url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Analyzing URL structure and running inference..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"url": target_url.strip()},
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )

                if response.status_code == 200:
                    data = response.json()

                    rf = data["models"]["random_forest"]
                    lr = data["models"]["logistic_regression"]
                    latency = data["latency_ms"]
                    agreement = data["agreement"]

                    st.markdown("---")

                    if not agreement:
                        st.warning("⚠️ The two models disagree on this URL -- treat the result with extra caution.")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("🌲 Random Forest")
                        if rf["is_phishing"]:
                            st.error("🚨 PHISHING DETECTED")
                        else:
                            st.success("✅ SAFE URL")
                        st.metric("Confidence", f"{rf['confidence_score']*100:.2f}%")

                    with col2:
                        st.subheader("📈 Logistic Regression")
                        if lr["is_phishing"]:
                            st.error("🚨 PHISHING DETECTED")
                        else:
                            st.success("✅ SAFE URL")
                        st.metric("Confidence", f"{lr['confidence_score']*100:.2f}%")

                    with col3:
                        st.subheader("⏱️ Performance")
                        st.metric("Total Inference Latency", f"{latency:.2f} ms")
                        st.metric("Model Agreement", "Yes" if agreement else "No")

                    # Feature Breakdown Table
                    st.subheader("🔎 Extracted Lexical & Structural Features")
                    raw_features = data["raw_features"]
                    feat_df = pd.DataFrame(list(raw_features.items()), columns=["Feature", "Extracted Value"])
                    feat_df["Used by models?"] = feat_df["Feature"].apply(
                        lambda f: "No (excluded -- dataset bias, see Discussion ch.)" if f in ("path_length", "count_slash") else "Yes"
                    )
                    st.dataframe(feat_df, use_container_width=True)

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Ensure `uvicorn src.api.app:app --host 127.0.0.1 --port 8000` is running.")
