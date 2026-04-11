# ==============================================================================
# Project: CyberShield Pro (Final Year Project)
# Developer: Aniket Patil
# Degree: BCA (Artificial Intelligence)
# Enrollment No: VGU23ONS2BCA0192
# University: Vivekananda Global University (VGU), Jaipur
# Year: 2026
# ==============================================================================


"""
CyberShield Pro ── VirusTotal v3 API Client
=============================================
Thin, stateless wrapper around VirusTotal's /urls endpoint.

Security
--------
  The API key is read EXCLUSIVELY from the environment variable VT_API_KEY.
  It is NEVER hardcoded in source.

Public surface
--------------
  VirusTotalClient.scan_url(url) → dict
  apply_vt_override(ml_result, vt_result) → dict
"""

import os
import base64
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

VT_BASE      = "https://www.virustotal.com/api/v3"
TIMEOUT      = 15   # seconds per request


class VirusTotalClient:
    """
    Thread-safe VirusTotal v3 client.
    One instance is created at app startup and reused for all requests.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.environ.get("VT_API_KEY", "")
        if not self._key:
            logger.warning("VT_API_KEY not set — VirusTotal lookups will be skipped.")
        self._session = requests.Session()
        self._session.headers.update({
            "x-apikey": self._key,
            "Accept":   "application/json",
        })

    # ── Public ──────────────────────────────────────────────────────────────

    def scan_url(self, url: str) -> dict:
        """
        Submit *url* to VirusTotal and return structured analysis results.

        Return schema
        -------------
        {
          available      : bool   – False when API key missing or call fails
          malicious      : int
          suspicious     : int
          harmless       : int
          undetected     : int
          total_engines  : int
          vt_link        : str    – permalink to the report GUI
          reputation     : int    – community score
          categories     : dict   – vendor category labels
          error          : str | None
        }
        """
        empty = self._empty()
        if not self._key:
            empty["error"] = "VT_API_KEY not configured"
            return empty

        try:
            url_id = self._encode(url)
            report = self._fetch_report(url_id)

            if report is None:                   # first-ever submission
                self._submit(url)
                time.sleep(3)
                report = self._fetch_report(url_id)
                if report is None:
                    empty["error"] = "VT report unavailable after submission"
                    return empty

            return self._parse(report)

        except requests.exceptions.Timeout:
            logger.error("VT timeout for %s", url)
            empty["error"] = "Request timed out"
            return empty
        except requests.exceptions.RequestException as exc:
            logger.error("VT network error: %s", exc)
            empty["error"] = str(exc)
            return empty
        except Exception as exc:          # noqa: BLE001
            logger.exception("VT unexpected error: %s", exc)
            empty["error"] = str(exc)
            return empty

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _encode(url: str) -> str:
        return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")

    def _submit(self, url: str) -> None:
        r = self._session.post(f"{VT_BASE}/urls", data={"url": url}, timeout=TIMEOUT)
        r.raise_for_status()

    def _fetch_report(self, url_id: str) -> Optional[dict]:
        r = self._session.get(f"{VT_BASE}/urls/{url_id}", timeout=TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _parse(report: dict) -> dict:
        attrs  = report.get("data", {}).get("attributes", {})
        stats  = attrs.get("last_analysis_stats", {})
        mal    = int(stats.get("malicious",  0))
        susp   = int(stats.get("suspicious", 0))
        harm   = int(stats.get("harmless",   0))
        undet  = int(stats.get("undetected", 0))
        total  = mal + susp + harm + undet
        uid    = report.get("data", {}).get("id", "")
        return {
            "available":     True,
            "malicious":     mal,
            "suspicious":    susp,
            "harmless":      harm,
            "undetected":    undet,
            "total_engines": total,
            "vt_link":       f"https://www.virustotal.com/gui/url/{uid}",
            "reputation":    int(attrs.get("reputation", 0)),
            "categories":    attrs.get("categories", {}),
            "error":         None,
        }

    @staticmethod
    def _empty() -> dict:
        return {
            "available": False, "malicious": 0, "suspicious": 0,
            "harmless": 0, "undetected": 0, "total_engines": 0,
            "vt_link": "", "reputation": 0, "categories": {}, "error": None,
        }


# =============================================================================
# Guardrail / override logic
# =============================================================================

def apply_vt_override(ml_result: dict, vt_result: dict) -> dict:
    """
    Cross-check ML prediction against VirusTotal and escalate when necessary.

    Override rules (strict, in priority order)
    -------------------------------------------
    Rule 1 — Critical Escalation
        ML = 'Safe'  AND  VT malicious ≥ 1
        → Final verdict: 'Malicious', risk: 'HIGH'  [MANDATORY per spec]

    Rule 2 — Soft Escalation
        ML = 'Safe'  AND  VT suspicious ≥ 2
        → Final verdict: 'Suspicious', risk: 'MEDIUM'

    Rule 3 — Confirmation (no change)
        ML = 'Malicious'  AND  VT malicious ≥ 1
        → Verdict kept, flagged as 'confirmed'

    Rule 4 — VT Offline
        VT not available → ML verdict kept, reason logged

    Returns
    -------
    {
      final_verdict   : str   'Safe' | 'Suspicious' | 'Malicious'
      final_risk      : str   'LOW'  | 'MEDIUM'     | 'HIGH'
      overridden      : bool
      override_reason : str
    }
    """
    ml_verdict    = ml_result.get("ml_verdict", "Safe")
    vt_available  = vt_result.get("available", False)
    vt_malicious  = vt_result.get("malicious", 0)
    vt_suspicious = vt_result.get("suspicious", 0)

    final_verdict  = ml_verdict
    final_risk     = ml_result.get("risk_level", "LOW")
    overridden     = False
    reason         = ""

    if not vt_available:
        reason = "VirusTotal unavailable — ML verdict applied."
        return dict(final_verdict=final_verdict, final_risk=final_risk,
                    overridden=overridden, override_reason=reason)

    # Rule 1 — Critical (MANDATORY)
    if vt_malicious >= 1 and ml_verdict == "Safe":
        final_verdict, final_risk = "Malicious", "HIGH"
        overridden = True
        reason = (f"GUARDRAIL: AI predicted Safe but VirusTotal flagged "
                  f"{vt_malicious} malicious engine(s) — verdict escalated to Malicious.")

    # Rule 2 — Soft escalation
    elif vt_suspicious >= 2 and ml_verdict == "Safe":
        final_verdict, final_risk = "Suspicious", "MEDIUM"
        overridden = True
        reason = (f"GUARDRAIL: AI predicted Safe but VirusTotal flagged "
                  f"{vt_suspicious} suspicious engine(s) — verdict escalated to Suspicious.")

    # Rule 3 — Confirmation
    elif ml_verdict == "Malicious" and vt_malicious >= 1:
        reason = (f"AI and VirusTotal agree: {vt_malicious} malicious engine(s) confirm verdict.")

    else:
        reason = (f"ML verdict ({ml_verdict}) accepted. "
                  f"VT reported {vt_malicious} malicious, {vt_suspicious} suspicious.")

    return dict(final_verdict=final_verdict, final_risk=final_risk,
                overridden=overridden, override_reason=reason)
