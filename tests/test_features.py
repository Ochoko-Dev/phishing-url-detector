import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features.extract_features import FeatureExtractor

def test_feature_extraction_structure():
    url = "http://equity-online-verify-ke.com/login"
    features = FeatureExtractor.extract(url)
    
    assert isinstance(features, dict)
    assert "url_length" in features
    assert "count_slash" in features
    assert "count_hyphens" in features
    assert features["url_length"] == len(url)

def test_feature_values_legitimate_url():
    url = "https://google.com"
    features = FeatureExtractor.extract(url)
    
    assert features["count_hyphens"] == 0
    assert features["count_at"] == 0

def test_feature_values_suspicious_url():
    url = "http://verify-account-update-bank.com/login/auth"
    features = FeatureExtractor.extract(url)
    
    assert features["count_hyphens"] >= 3
    assert features["count_slash"] >= 4
