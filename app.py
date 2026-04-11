# ==============================================================================
# Project: CyberShield Pro (Final Year Project)
# Developer: Aniket Patil
# Degree: BCA (Artificial Intelligence)
# Enrollment No: VGU23ONS2BCA0192
# University: Vivekananda Global University (VGU), Jaipur
# Year: 2026
# ==============================================================================


"""
CyberShield Pro ── Flask Application
=======================================
Run:
    python app.py                           (development)
    gunicorn app:app -b 0.0.0.0:5000        (production)

Route map
---------
  Public
    GET  /                       → scanner UI
    POST /api/scan               → dual-layer URL analysis
    GET  /api/health             → liveness probe

  Admin (session-protected)
    GET  /admin/login            → login form
    POST /admin/login            → authenticate
    GET  /admin/logout           → destroy session
    GET  /admin/dashboard        → admin SPA
    GET  /api/admin/stats        → aggregated stats + chart data
    GET  /api/admin/scans        → paginated scan table
    POST /api/admin/retrain      → background model retrain
    DELETE /api/admin/scan/<id>  → delete a single record
"""

import os
import json
import logging
import threading
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, g,
)
from dotenv import load_dotenv

# Load .env BEFORE importing our modules so env-vars are available
load_dotenv()

from core.ml_engine  import smart_predict, retrain_from_db, model_info
from core.vt_client  import VirusTotalClient, apply_vt_override

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-please")

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "data" / "cybershield.db"

# Global retrain state (thread-safe via GIL for simple flag)
_retrain_status = {"running": False, "last_result": None, "last_error": None}


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db: db.close()


def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS scans (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                url             TEXT    NOT NULL,
                scanner_ip      TEXT,
                ml_score        REAL    NOT NULL DEFAULT 0,
                ml_verdict      TEXT    NOT NULL DEFAULT 'Safe',
                vt_malicious    INTEGER NOT NULL DEFAULT 0,
                vt_suspicious   INTEGER NOT NULL DEFAULT 0,
                vt_harmless     INTEGER NOT NULL DEFAULT 0,
                vt_undetected   INTEGER NOT NULL DEFAULT 0,
                vt_total        INTEGER NOT NULL DEFAULT 0,
                final_verdict   TEXT    NOT NULL DEFAULT 'Safe',
                final_risk      TEXT    NOT NULL DEFAULT 'LOW',
                overridden      INTEGER NOT NULL DEFAULT 0,
                override_reason TEXT,
                scan_method     TEXT    DEFAULT 'heuristic',
                timestamp       TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrain_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_at TEXT NOT NULL,
                status      TEXT NOT NULL,
                accuracy    REAL,
                n_samples   INTEGER,
                error       TEXT
            );
        """)
        db.commit()
    logger.info("Database ready at %s", DB_PATH)


# ── VirusTotal client singleton ───────────────────────────────────────────────
vt_client = VirusTotalClient()


# =============================================================================
# Auth helpers
# =============================================================================

ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "cybershield2024")


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


def api_login_required(f):
    """For JSON API admin endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# Public routes
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status":        "ok",
        "vt_configured": bool(os.environ.get("VT_API_KEY")),
        "model_info":    model_info(),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/scan", methods=["POST"])
def scan_url():
    """
    Dual-layer URL analysis endpoint.

    Body: { "url": "https://example.com" }
    """
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url field is required"}), 400

    # Normalise scheme
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    scanner_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    try:
        # Layer 1 — ML
        ml   = smart_predict(url)
        # Layer 2 — VirusTotal
        vt   = vt_client.scan_url(url)
        # Guardrail cross-check
        over = apply_vt_override(ml, vt)

        ts = datetime.now(timezone.utc).isoformat()

        db = get_db()
        db.execute("""
            INSERT INTO scans
              (url, scanner_ip, ml_score, ml_verdict,
               vt_malicious, vt_suspicious, vt_harmless, vt_undetected, vt_total,
               final_verdict, final_risk, overridden, override_reason, scan_method, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            url, scanner_ip,
            ml["ml_score"], ml["ml_verdict"],
            vt["malicious"], vt["suspicious"], vt["harmless"], vt["undetected"], vt["total_engines"],
            over["final_verdict"], over["final_risk"],
            int(over["overridden"]), over["override_reason"],
            ml.get("method", "heuristic"), ts,
        ))
        db.commit()

        logger.info("SCAN %s → %s (ml=%.1f%% vt_mal=%d override=%s)",
                    url, over["final_verdict"], ml["ml_score"],
                    vt["malicious"], over["overridden"])

        return jsonify({
            "url":             url,
            "scanner_ip":      scanner_ip,
            "ml_score":        ml["ml_score"],
            "ml_verdict":      ml["ml_verdict"],
            "risk_level":      ml["risk_level"],
            "class_probs":     ml["class_probs"],
            "features_used":   ml["features_used"],
            "vt":              vt,
            "final_verdict":   over["final_verdict"],
            "final_risk":      over["final_risk"],
            "overridden":      over["overridden"],
            "override_reason": over["override_reason"],
            "scan_method":     ml.get("method", "heuristic"),
            "timestamp":       ts,
        })

    except Exception as exc:      # noqa: BLE001
        logger.exception("Scan error for %s: %s", url, exc)
        return jsonify({"error": str(exc)}), 500


# =============================================================================
# Admin — auth
# =============================================================================

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        uname = request.form.get("username", "").strip()
        pwd   = request.form.get("password", "")
        if uname == ADMIN_USER and pwd == ADMIN_PASS:
            session["admin_logged_in"] = True
            session["admin_user"]      = uname
            logger.info("Admin login: %s from %s", uname, request.remote_addr)
            return redirect(url_for("admin_dashboard"))
        error = "Invalid credentials. Please try again."
        logger.warning("Failed admin login attempt from %s", request.remote_addr)
    return render_template("login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    return render_template("admin.html", admin_user=session.get("admin_user", "admin"))


# =============================================================================
# Admin — JSON API
# =============================================================================

@app.route("/api/admin/stats")
@api_login_required
def admin_stats():
    """Aggregate stats + time-series data for Chart.js."""
    db  = get_db()

    # ── Totals ─────────────────────────────────────────────────────────────
    row = db.execute("""
        SELECT
            COUNT(*)                               AS total,
            SUM(final_verdict = 'Safe')            AS safe_cnt,
            SUM(final_verdict = 'Suspicious')      AS susp_cnt,
            SUM(final_verdict = 'Malicious')       AS mal_cnt,
            SUM(overridden = 1)                    AS override_cnt,
            AVG(ml_score)                          AS avg_score
        FROM scans
    """).fetchone()

    # ── Daily activity (last 14 days) ──────────────────────────────────────
    daily = db.execute("""
        SELECT
            DATE(timestamp) AS day,
            COUNT(*)        AS total,
            SUM(final_verdict = 'Malicious') AS malicious
        FROM scans
        WHERE timestamp >= DATE('now', '-14 days')
        GROUP BY day
        ORDER BY day ASC
    """).fetchall()

    # ── Method breakdown ───────────────────────────────────────────────────
    methods = db.execute("""
        SELECT scan_method, COUNT(*) AS cnt FROM scans GROUP BY scan_method
    """).fetchall()

    # ── Risk breakdown ─────────────────────────────────────────────────────
    risks = db.execute("""
        SELECT final_risk, COUNT(*) AS cnt FROM scans GROUP BY final_risk
    """).fetchall()

    # ── Model info ─────────────────────────────────────────────────────────
    m_info  = model_info()
    retrain = db.execute("""
        SELECT * FROM retrain_log ORDER BY id DESC LIMIT 5
    """).fetchall()

    return jsonify({
        "totals": {
            "total":     row["total"]       or 0,
            "safe":      row["safe_cnt"]    or 0,
            "suspicious":row["susp_cnt"]    or 0,
            "malicious": row["mal_cnt"]     or 0,
            "overrides": row["override_cnt"]or 0,
            "avg_score": round(row["avg_score"] or 0, 1),
        },
        "daily_activity": [dict(d) for d in daily],
        "method_breakdown": [dict(m) for m in methods],
        "risk_breakdown":   [dict(r) for r in risks],
        "model_info":       m_info,
        "retrain_log":      [dict(r) for r in retrain],
        "retrain_status":   _retrain_status,
    })


@app.route("/api/admin/scans")
@api_login_required
def admin_scans():
    """Paginated scan records for the data table."""
    page     = max(int(request.args.get("page", 1)), 1)
    per_page = min(int(request.args.get("per_page", 25)), 100)
    search   = request.args.get("q", "").strip()
    verdict  = request.args.get("verdict", "").strip()
    offset   = (page - 1) * per_page

    where_clauses = []
    params: list = []

    if search:
        where_clauses.append("url LIKE ?")
        params.append(f"%{search}%")
    if verdict:
        where_clauses.append("final_verdict = ?")
        params.append(verdict)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    db    = get_db()
    total = db.execute(f"SELECT COUNT(*) FROM scans {where_sql}", params).fetchone()[0]
    rows  = db.execute(
        f"""SELECT id, url, scanner_ip, ml_score, vt_malicious,
                   final_verdict, final_risk, overridden, scan_method, timestamp
            FROM scans {where_sql}
            ORDER BY id DESC LIMIT ? OFFSET ?""",
        params + [per_page, offset],
    ).fetchall()

    return jsonify({
        "scans":      [dict(r) for r in rows],
        "total":      total,
        "page":       page,
        "per_page":   per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    })


@app.route("/api/admin/scan/<int:scan_id>", methods=["DELETE"])
@api_login_required
def delete_scan(scan_id):
    db = get_db()
    db.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    db.commit()
    return jsonify({"deleted": scan_id})


@app.route("/api/admin/retrain", methods=["POST"])
@api_login_required
def trigger_retrain():
    """
    Fire a background thread to retrain the model from the database.
    Returns immediately; poll /api/admin/stats for retrain_status.
    """
    global _retrain_status

    if _retrain_status["running"]:
        return jsonify({"message": "Retrain already in progress.", "status": "running"}), 409

    def _retrain_worker():
        global _retrain_status
        _retrain_status = {"running": True, "last_result": None, "last_error": None}
        try:
            result = retrain_from_db(str(DB_PATH))
            _retrain_status["last_result"] = result
            # Log to DB
            with sqlite3.connect(str(DB_PATH)) as conn:
                conn.execute("""
                    INSERT INTO retrain_log (triggered_at, status, accuracy, n_samples)
                    VALUES (?,?,?,?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    "success",
                    result.get("accuracy"),
                    result.get("n_train", 0) + result.get("n_test", 0),
                ))
            logger.info("Background retrain complete. accuracy=%.4f", result["accuracy"])
        except Exception as exc:   # noqa: BLE001
            _retrain_status["last_error"] = str(exc)
            try:
                with sqlite3.connect(str(DB_PATH)) as conn:
                    conn.execute("""
                        INSERT INTO retrain_log (triggered_at, status, error)
                        VALUES (?,?,?)
                    """, (datetime.now(timezone.utc).isoformat(), "error", str(exc)))
            except Exception:
                pass
            logger.error("Background retrain failed: %s", exc)
        finally:
            _retrain_status["running"] = False

    t = threading.Thread(target=_retrain_worker, daemon=True)
    t.start()

    return jsonify({
        "message": "Retrain started in background. Poll /api/admin/stats for status.",
        "status":  "started",
    })


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    init_db()
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port  = int(os.environ.get("PORT", 5000))
    logger.info("CyberShield Pro → http://localhost:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
