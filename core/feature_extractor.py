# ==============================================================================
# Project: CyberShield Pro (Final Year Project)
# Developer: Aniket Patil
# Degree: BCA (Artificial Intelligence)
# Enrollment No: VGU23ONS2BCA0192
# University: Vivekananda Global University (VGU), Jaipur
# Year: 2026
# ==============================================================================


"""
CyberShield Pro ── Feature Extractor
======================================
Centralised URL → feature-vector pipeline.
Used identically during training and inference to prevent train/serve skew.

Features (12 total)
-------------------
 0  url_length           Total character count of the URL
 1  has_ip               1 if hostname is an IPv4 address
 2  has_at_symbol        1 if '@' present (credential injection trick)
 3  has_question_mark    1 if '?' present
 4  has_hyphen           1 if '-' present in hostname
 5  has_equals           1 if '=' present
 6  subdomain_count      Number of labels beyond domain + TLD
 7  is_https             1 if scheme is 'https'
 8  path_depth           Count of '/' in URL path
 9  digit_letter_ratio   Digits / letters in hostname (0 if no letters)
10  special_char_count   Count of { %, #, &, ~, ;, ! } in URL
11  url_entropy          Shannon entropy of the full URL string
"""

import re
import math
from urllib.parse import urlparse


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(url: str) -> list[float]:
    """Return a 12-element float feature vector for *url*."""
    parsed   = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    path     = parsed.path     or ""

    # 0 – url_length
    url_length = len(url)

    # 1 – has_ip
    _ipv4 = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
    has_ip = 1.0 if _ipv4.match(hostname) else 0.0

    # 2-5 – character flags
    has_at            = 1.0 if "@" in url      else 0.0
    has_question_mark = 1.0 if "?" in url      else 0.0
    has_hyphen        = 1.0 if "-" in hostname else 0.0
    has_equals        = 1.0 if "=" in url      else 0.0

    # 6 – subdomain_count
    labels          = [l for l in hostname.split(".") if l]
    subdomain_count = float(max(len(labels) - 2, 0))

    # 7 – is_https
    is_https = 1.0 if parsed.scheme == "https" else 0.0

    # 8 – path_depth
    path_depth = float(path.count("/"))

    # 9 – digit_letter_ratio
    digits  = sum(c.isdigit() for c in hostname)
    letters = sum(c.isalpha() for c in hostname)
    digit_letter_ratio = digits / letters if letters else 0.0

    # 10 – special_char_count
    special_set        = set("%#&~;!")
    special_char_count = float(sum(c in special_set for c in url))

    # 11 – url_entropy
    url_entropy = _shannon_entropy(url)

    return [
        float(url_length), has_ip, has_at, has_question_mark,
        has_hyphen, has_equals, subdomain_count, is_https,
        path_depth, digit_letter_ratio, special_char_count, url_entropy,
    ]


def feature_names() -> list[str]:
    """Canonical ordered list of feature names (matches extract_features)."""
    return [
        "url_length", "has_ip", "has_at_symbol", "has_question_mark",
        "has_hyphen", "has_equals", "subdomain_count", "is_https",
        "path_depth", "digit_letter_ratio", "special_char_count", "url_entropy",
    ]


# ── Internals ─────────────────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())
