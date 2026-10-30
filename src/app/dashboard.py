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

API_URL = "http://127.0.0.1:8000/predict"

st.title("🛡️ Phishing URL Threat Detection Engine")
st.markdown("Real-time lexical and structural URL analysis powered by Machine Learning.")

# Sidebar Controls
st.sidebar.header("Configuration & API Status")
try:
    health_check = requests.get("http://127.0.0.1:8000/", timeout=2)
    if health_check.status_code == 200:
        st.sidebar.success("FastAPI Backend: ONLINE")
    else:
        st.sidebar.warning("FastAPI Backend: UNHEALTHY")
except Exception:
    st.sidebar.error("FastAPI Backend: OFFLINE")

st.sidebar.info("Framework Architecture:\n- Backend: FastAPI\n- ML Models: XGBoost / Random Forest\n- Latency Target: < 2.0s")

# Main Interface
target_url = st.text_input("Enter URL to scan:", placeholder="e.g. http://equity-online-verify-ke.com/login")

if st.button("Scan URL", type="primary"):
    if not target_url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Analyzing URL structure and running inference..."):
            try:
                # 1. Query FastAPI Backend
                response = requests.post(
                    API_URL,
                    json={"url": target_url.strip()},
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    is_phishing = data["is_phishing"]
                    confidence = data["confidence_score"] * 100
                    latency = data["latency_ms"]
                    model_used = data["model_used"]

                    st.markdown("---")
                    
                    # Top Metrics Banner
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if is_phishing:
                            st.error("🚨 RESULT: PHISHING DETECTED")
                        else:
                            st.success("✅ RESULT: SAFE URL")
                            
                    with col2:
                        st.metric("Confidence Level", f"{confidence:.2f}%")
                        
                    with col3:
                        st.metric("Inference Latency", f"{latency:.2f} ms", delta_color="inverse")

                    # Feature Breakdown Table
                    st.subheader("📊 Extracted Lexical & Structural Features")
                    extracted_features = FeatureExtractor.extract(target_url.strip())
                    
                    feat_df = pd.DataFrame([extracted_features]).T.reset_index()
                    feat_df.columns = ["Feature Metric", "Extracted Value"]
                    
                    st.dataframe(feat_df, use_container_width=True)
                    
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Ensure `uvicorn src.api.app:app` is running on port 8000.")
