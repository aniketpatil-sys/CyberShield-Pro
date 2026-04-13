# ==============================================================================
# Project: CyberShield Pro (Final Year Project)
# Developer: Aniket Patil
# Degree: BCA (Artificial Intelligence)
# Enrollment No: VGU23ONS2BCA0192
# University: Vivekananda Global University (VGU), Jaipur
# Year: 2026
# ==============================================================================

"""
CyberShield Pro ── ML Engine
==============================
Handles RandomForestClassifier training, persistence, and inference.

Public surface
--------------
  smart_predict(url)          → dict   (uses RF model or heuristic fallback)
  train_from_csv(csv_path)    → dict   (training summary)
  retrain_from_db(db_path)    → dict   (on-demand retrain from SQLite)
  model_info()                → dict   (metadata about loaded model)
"""

import os
import pickle
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from core.feature_extractor import extract_features, feature_names

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH  = _ROOT / "models" / "rf_model.pkl"
SCALER_PATH = _ROOT / "models" / "scaler.pkl"
META_PATH   = _ROOT / "models" / "meta.pkl"

# ── Thresholds ────────────────────────────────────────────────────────────────
THR_LOW    = 0.35   # threat_score < 35 % → LOW  / Safe
THR_MEDIUM = 0.65   # threat_score < 65 % → MED  / Suspicious
                    # threat_score ≥ 65 % → HIGH / Malicious

# ── In-process model cache ────────────────────────────────────────────────────
_model  = None
_scaler = None
_meta   = {}


# =============================================================================
# Training
# =============================================================================

def _fit_model(X_raw: np.ndarray, y: np.ndarray) -> dict:
    """Core fit routine shared by CSV and DB training paths."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_split=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_train)

    y_pred   = clf.predict(X_te)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=["safe", "suspicious", "malicious"],
        zero_division=0,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(clf,    f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    meta = {
        "trained_at":          datetime.now(timezone.utc).isoformat(),
        "n_train":             int(len(X_train)),
        "n_test":              int(len(X_test)),
        "accuracy":            round(float(accuracy), 4),
        "feature_importances": dict(zip(feature_names(), clf.feature_importances_.tolist())),
        "classes":             clf.classes_.tolist(),
    }
    with open(META_PATH, "wb") as f: pickle.dump(meta, f)

    # Invalidate in-process cache so next predict() reloads fresh artifacts
    global _model, _scaler, _meta
    _model = _scaler = None
    _meta  = meta

    logger.info("Model trained. accuracy=%.4f  n=%d", accuracy, len(X_raw))
    logger.info("\n%s", report)
    return {"accuracy": accuracy, "n_train": len(X_train), "n_test": len(X_test),
            "feature_importances": meta["feature_importances"], "report": report}


def train_from_csv(csv_path: str) -> dict:
    """Train from a labelled CSV (columns: url, label)."""
    import pandas as pd
    label_map = {"safe": 0, "suspicious": 1, "malicious": 2}
    df = pd.read_csv(csv_path)
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map(label_map)
    df.dropna(subset=["url", "label"], inplace=True)
    X = np.array([extract_features(u) for u in df["url"]])
    y = df["label"].astype(int).values
    return _fit_model(X, y)


def retrain_from_db(db_path: str) -> dict:
    """
    Retrain using records stored in the SQLite scan database.
    Rows are labelled by their final_verdict column.
    """
    label_map = {"Safe": 0, "Suspicious": 1, "Malicious": 2}
    conn  = sqlite3.connect(db_path)
    rows  = conn.execute(
        "SELECT url, final_verdict FROM scans WHERE final_verdict IS NOT NULL"
    ).fetchall()
    conn.close()

    if len(rows) < 10:
        raise ValueError(
            f"Only {len(rows)} labelled records found in DB. "
            "Need at least 10 to retrain."
        )

    urls    = [r[0] for r in rows]
    labels  = [label_map.get(r[1], 0) for r in rows]
    X       = np.array([extract_features(u) for u in urls])
    y       = np.array(labels)
    return _fit_model(X, y)


# =============================================================================
# Inference
# =============================================================================

def _load_artifacts():
    global _model, _scaler, _meta
    if _model is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError("ML model not found. Run train_model.py first.")
        with open(MODEL_PATH,  "rb") as f: _model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
        if META_PATH.exists():
            with open(META_PATH, "rb") as f: _meta = pickle.load(f)
        logger.info("ML artifacts loaded from disk.")
    return _model, _scaler


def predict(url: str) -> dict:
    """RF-model prediction. Raises FileNotFoundError if model absent."""
    model, scaler = _load_artifacts()
    feats  = extract_features(url)
    scaled = scaler.transform([feats])
    probs  = model.predict_proba(scaled)[0]          # [p_safe, p_susp, p_mal]

    # Weighted threat score: suspicious × 0.4  +  malicious × 0.6
    threat = float(probs[1]) * 0.4 + float(probs[2]) * 0.6
    score  = round(threat * 100, 1)

    if threat < THR_LOW:
        risk, verdict = "LOW",    "Safe"
    elif threat < THR_MEDIUM:
        risk, verdict = "MEDIUM", "Suspicious"
    else:
        risk, verdict = "HIGH",   "Malicious"

    return {
        "ml_score":     score,
        "risk_level":   risk,
        "ml_verdict":   verdict,
        "class_probs":  {
            "safe":        round(float(probs[0]), 4),
            "suspicious":  round(float(probs[1]), 4),
            "malicious":   round(float(probs[2]), 4),
        },
        "features_used": dict(zip(feature_names(), feats)),
        "method":        "ml_model",
    }


def predict_heuristic(url: str) -> dict:
    """Rule-based fallback — no trained model required."""
    feats = extract_features(url)
    fd    = dict(zip(feature_names(), feats))

    score = 0.0
    if fd["has_ip"]:                   score += 30
    if fd["has_at_symbol"]:            score += 25
    if not fd["is_https"]:             score += 15
    if fd["subdomain_count"] > 3:      score += 15
    if fd["url_length"] > 100:         score += 10
    if fd["digit_letter_ratio"] > 0.5: score += 10
    if fd["url_entropy"] > 4.5:        score += 10
    if fd["special_char_count"] > 5:   score += 10
    score = min(score, 100.0)

    if score < 35:    risk, verdict = "LOW",    "Safe"
    elif score < 65:  risk, verdict = "MEDIUM", "Suspicious"
    else:             risk, verdict = "HIGH",   "Malicious"

    return {
        "ml_score":     round(score, 1),
        "risk_level":   risk,
        "ml_verdict":   verdict,
        "class_probs":  {"safe": 0.0, "suspicious": 0.0, "malicious": 0.0},
        "features_used": fd,
        "method":       "heuristic",
    }


def smart_predict(url: str) -> dict:
    """Try RF model; fall back to heuristic silently if model absent."""
    try:
        return predict(url)
    except FileNotFoundError:
        logger.warning("Model unavailable — using heuristic fallback.")
        return predict_heuristic(url)


def model_info() -> dict:
    """Return metadata about the currently persisted model."""
    if META_PATH.exists() and not _meta:
        with open(META_PATH, "rb") as f:
            m = pickle.load(f)
    else:
        m = _meta
    return {
        "model_exists":  MODEL_PATH.exists(),
        "trained_at":    m.get("trained_at", "N/A"),
        "accuracy":      m.get("accuracy",   None),
        "n_train":       m.get("n_train",    None),
        "top_features":  sorted(
            m.get("feature_importances", {}).items(),
            key=lambda x: x[1], reverse=True
        )[:5] if m.get("feature_importances") else [],
    }
