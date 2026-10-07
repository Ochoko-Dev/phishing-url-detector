import re
import math
from urllib.parse import urlparse

class FeatureExtractor:
    """
    Extracts lexical, structural, and statistical features from raw URL strings.
    """

    @staticmethod
    def calculate_entropy(text):
        if not text:
            return 0.0
        entropy = 0.0
        length = len(text)
        frequencies = {char: text.count(char) for char in set(text)}
        for count in frequencies.values():
            p = count / length
            entropy -= p * math.log2(p)
        return round(entropy, 4)

    @classmethod
    def extract(cls, url):
        url_str = str(url).strip()
        
        # Ensure scheme for proper parsing
        if not url_str.startswith(('http://', 'https://')):
            parsed_url = urlparse('http://' + url_str)
        else:
            parsed_url = urlparse(url_str)
            
        domain = parsed_url.netloc or parsed_url.path.split('/')[0]
        path = parsed_url.path
        query = parsed_url.query

        # Basic Structural Features
        url_length = len(url_str)
        domain_length = len(domain)
        path_length = len(path)
        
        # Character Counts
        count_dots = url_str.count('.')
        count_hyphens = url_str.count('-')
        count_at = url_str.count('@')
        count_question = url_str.count('?')
        count_equal = url_str.count('=')
        count_slash = url_str.count('/')
        count_digits = sum(c.isdigit() for c in url_str)
        
        # Ratios & Metrics
        digit_ratio = round(count_digits / url_length, 4) if url_length > 0 else 0.0
        domain_entropy = cls.calculate_entropy(domain)
        
        # IP Address Detection
        ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        host_no_port = domain.split(':')[0]
        is_ip = 1 if re.match(ip_pattern, host_no_port) else 0

        # Subdomain Depth Count
        subdomain_count = len(domain.split('.')) - 2 if not is_ip else 0
        if subdomain_count < 0:
            subdomain_count = 0

        # HTTPS Protocol Indicator
        uses_https = 1 if url_str.startswith('https://') else 0

        # Suspicious Keyword Detection
        suspicious_keywords = ['login', 'verify', 'update', 'account', 'banking', 'secure', 'vooma', 'pesa', 'unblock']
        has_suspicious_keyword = 1 if any(kw in url_str.lower() for kw in suspicious_keywords) else 0

        return {
            'url_length': url_length,
            'domain_length': domain_length,
            'path_length': path_length,
            'count_dots': count_dots,
            'count_hyphens': count_hyphens,
            'count_at': count_at,
            'count_question': count_question,
            'count_equal': count_equal,
            'count_slash': count_slash,
            'digit_ratio': digit_ratio,
            'domain_entropy': domain_entropy,
            'is_ip': is_ip,
            'subdomain_count': subdomain_count,
            'uses_https': uses_https,
            'has_suspicious_keyword': has_suspicious_keyword
        }
