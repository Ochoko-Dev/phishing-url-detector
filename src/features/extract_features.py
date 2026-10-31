import re
import math
from urllib.parse import urlparse

try:
    import Levenshtein
    def _lev(a, b):
        return Levenshtein.distance(a, b)
except ImportError:
    def _lev(a, b):
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return len(a)
        previous_row = range(len(b) + 1)
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


class FeatureExtractor:
    """
    Extracts lexical, structural, and brand-impersonation features
    from raw URL strings. No network calls, no external lookups.
    """

    KENYAN_BRANDS = [
        "equity", "kcb", "coop", "cooperative", "absa", "mpesa",
        "safaricom", "ncba", "stanbic", "dtb", "familybank"
    ]

    # Generic TLD/suffix fragments excluded from per-label brand comparison
    # so short tokens like 'co' or 'com' don't accidentally collide with
    # short brand names like 'coop'.
    GENERIC_LABELS = {
        "com", "co", "ke", "org", "net", "info", "xyz", "top", "go",
        "io", "biz", "online", "app"
    }

    # Canonical official domains, used only to flag a brand-name mention
    # that does NOT match the real institution's domain — the strongest
    # single impersonation signal available without a network lookup.
    OFFICIAL_DOMAINS = {
        "equitygroupholdings.com", "equityonline.equitygroupholdings.com",
        "kcbgroup.com", "co-opbank.co.ke", "i-bank.co-opbank.co.ke",
        "absabank.co.ke", "ncbagroup.com", "familybank.co.ke",
        "stanbicbank.co.ke", "dtbafrica.com", "dtbk.dtbafrica.com", "safaricom.co.ke"
    }

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
    def brand_distance(cls, domain):
        """Minimum Levenshtein distance between any meaningful (non-generic)
        dot-separated label of the domain and a known brand name. Comparing
        per-label, with generic suffixes excluded, catches a brand hidden
        inside a longer spoofed subdomain without colliding on TLD noise."""
        domain_clean = domain.lower().split(':')[0]
        labels = [lbl for lbl in domain_clean.split('.')
                  if lbl and lbl not in cls.GENERIC_LABELS]
        if not labels:
            return 99
        return min(_lev(lbl, brand) for lbl in labels for brand in cls.KENYAN_BRANDS)

    @classmethod
    def brand_in_domain(cls, domain):
        """1 if any known brand name appears verbatim anywhere in the
        domain string, regardless of position."""
        domain_clean = domain.lower()
        return 1 if any(brand in domain_clean for brand in cls.KENYAN_BRANDS) else 0

    @classmethod
    def brand_domain_mismatch(cls, domain):
        """1 if a brand name is present but the domain is NOT the known
        official domain for that brand -- the strongest single
        impersonation signal: 'mentions a bank, isn't the bank'."""
        domain_clean = domain.lower().split(':')[0]
        if domain_clean.startswith('www.'):
            domain_clean = domain_clean[4:]
        if not cls.brand_in_domain(domain_clean):
            return 0
        return 0 if domain_clean in cls.OFFICIAL_DOMAINS else 1

    @classmethod
    def extract_features(cls, url):
        url_str = str(url).strip()

        if not url_str.startswith(('http://', 'https://')):
            parsed_url = urlparse('http://' + url_str)
        else:
            parsed_url = urlparse(url_str)

        domain = parsed_url.netloc or parsed_url.path.split('/')[0]
        path = parsed_url.path

        url_length = len(url_str)
        domain_length = len(domain)
        path_length = len(path)

        count_dots = url_str.count('.')
        count_hyphens = url_str.count('-')
        count_at = url_str.count('@')
        count_question = url_str.count('?')
        count_equal = url_str.count('=')
        count_slash = url_str.count('/')
        count_digits = sum(c.isdigit() for c in url_str)

        digit_ratio = round(count_digits / url_length, 4) if url_length > 0 else 0.0
        domain_entropy = cls.calculate_entropy(domain)

        ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        host_no_port = domain.split(':')[0]
        is_ip = 1 if re.match(ip_pattern, host_no_port) else 0

        subdomain_count = len(domain.split('.')) - 2 if not is_ip else 0
        if subdomain_count < 0:
            subdomain_count = 0

        uses_https = 1 if url_str.startswith('https://') else 0

        suspicious_keywords = ['login', 'verify', 'update', 'account', 'banking',
                                'secure', 'vooma', 'pesa', 'unblock', 'portal']
        has_suspicious_keyword = 1 if any(kw in url_str.lower() for kw in suspicious_keywords) else 0

        min_brand_distance = cls.brand_distance(host_no_port)
        brand_present = cls.brand_in_domain(host_no_port)
        brand_mismatch = cls.brand_domain_mismatch(host_no_port)

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
            'has_suspicious_keyword': has_suspicious_keyword,
            'min_brand_distance': min_brand_distance,
            'brand_present': brand_present,
            'brand_domain_mismatch': brand_mismatch
        }
