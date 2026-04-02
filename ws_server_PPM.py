"""
ws_server.py — Exotel Outbound Realtime LIC Agent + Call Logger + MCP tool-calls

Environment variables expected:

  PORT=10000 (Render)
  LOG_LEVEL=INFO or DEBUG

  EXOTEL_SID       gouravnxmx1
  EXOTEL_TOKEN     your token
  EXO_SUBDOMAIN    api or api.in  (not used in new outbound helper, but kept for compatibility)
  EXO_CALLER_ID    your Exophone, e.g. 02248904368

  # For outbound flow URL (from Exotel support):
  #   http://my.exotel.com/gouravnxmx1/exoml/start_voice/1077390
  EXOTEL_FLOW_URL  (optional; falls back to the above if not set)

  OPENAI_API_KEY or OpenAI_Key or OPENAI_KEY
  OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview (recommended)

  PUBLIC_BASE_URL  e.g. openai-exotel-sales-prediction.onrender.com
  DB_PATH=/tmp/call_logs.db   (or /data/call_logs.db if you have persistent disk)

  LIC_CRM_MCP_BASE_URL=https://lic-crm-mcp.onrender.com    (MCP server; we call /test-save)
"""

import asyncio
import base64
import json
import logging
from fastapi.responses import FileResponse
import os
import sqlite3
import time
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Any, List

import audioop
import httpx
from aiohttp import ClientSession, WSMsgType
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import sqlite3
import re
import io

import numpy as np
import pandas as pd
from fastapi import Query
from fastapi.responses import StreamingResponse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ppm.storage import (
        init_ppm_db,
        save_ppm_decision,
        save_ppm_outcome,
        save_ppm_debug_event,
        get_debug_events,
    )
except ModuleNotFoundError:
    init_ppm_db = None
    save_ppm_decision = None
    save_ppm_outcome = None
    save_ppm_debug_event = None
    get_debug_events = None

# Additive safety fallbacks so optional PPM storage/logging never crashes live call flow.
# This preserves existing variable names and only replaces missing / non-callable imports with no-op functions.
try:
    _ppm_init_db_callable = callable(init_ppm_db)
except Exception:
    _ppm_init_db_callable = False
if not _ppm_init_db_callable:
    def init_ppm_db(*args, **kwargs):
        return None

try:
    _ppm_save_decision_callable = callable(save_ppm_decision)
except Exception:
    _ppm_save_decision_callable = False
if not _ppm_save_decision_callable:
    def save_ppm_decision(*args, **kwargs):
        return None

try:
    _ppm_save_outcome_callable = callable(save_ppm_outcome)
except Exception:
    _ppm_save_outcome_callable = False
if not _ppm_save_outcome_callable:
    def save_ppm_outcome(*args, **kwargs):
        return None

try:
    _ppm_save_debug_event_callable = callable(save_ppm_debug_event)
except Exception:
    _ppm_save_debug_event_callable = False
if not _ppm_save_debug_event_callable:
    def save_ppm_debug_event(*args, **kwargs):
        return None

try:
    _ppm_get_debug_events_callable = callable(get_debug_events)
except Exception:
    _ppm_get_debug_events_callable = False
if not _ppm_get_debug_events_callable:
    def get_debug_events(*args, **kwargs):
        return []



# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("ws_server")

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

# Preferred Exotel env var names (DO NOT CHANGE)
EXO_SID = os.getenv("EXO_SID", "")
EXO_API_KEY = os.getenv("EXO_API_KEY", "")
EXO_API_TOKEN = os.getenv("EXO_API_TOKEN", "")
EXO_FLOW_ID = os.getenv("EXO_FLOW_ID", "")
EXO_CALLER_ID = os.getenv("EXO_CALLER_ID", "")

# Exotel outbound flow URL. If not explicitly set, build from EXO_SID + EXO_FLOW_ID when possible.
EXOTEL_FLOW_URL = os.getenv("EXOTEL_FLOW_URL", "").strip()
if not EXOTEL_FLOW_URL:
    if EXO_API_KEY  and EXO_FLOW_ID:
        EXOTEL_FLOW_URL = f"http://my.exotel.com/{EXO_SID}/exoml/start_voice/{EXO_FLOW_ID}"
    else:
        EXOTEL_FLOW_URL = "http://my.exotel.com/gouravnxmx1/exoml/start_voice/1077390"

# Backward-compatible aliases used elsewhere in code (keep variable names; map to EXO_*).
EXOTEL_SID = EXO_API_KEY
EXOTEL_TOKEN = EXO_API_TOKEN

EXO_SUBDOMAIN = os.getenv("EXO_SUBDOMAIN", "api")  # kept for compatibility

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OpenAI_Key")
    or os.getenv("OPENAI_KEY", "")
)
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

LIC_CRM_MCP_BASE_URL = os.getenv("LIC_CRM_MCP_BASE_URL", "").rstrip("/")

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip()  # no protocol

DB_PATH = os.getenv("DB_PATH", "/tmp/call_logs.db")
# =========================
# ADDITIVE VOICE CONTROL
# =========================
VOICE_AUTO_SPEAK_DELAY_MS = int(os.getenv("VOICE_AUTO_SPEAK_DELAY_MS", "800"))
VOICE_REENGAGE_DELAY_MS = int(os.getenv("VOICE_REENGAGE_DELAY_MS", "2500"))
VOICE_ENABLE_START_NUDGE = os.getenv("VOICE_ENABLE_START_NUDGE", "1") == "1"

def public_url(path: str) -> str:
    host = PUBLIC_BASE_URL
    if not host:
        # This is only used when Exotel calls us, so PUBLIC_BASE_URL really
        # should be set to your Render hostname.
        logger.warning("PUBLIC_BASE_URL is not set; using localhost (dev only).")
        host = "localhost:10000"
    path = path.lstrip("/")
    return f"https://{host}/{path}"


# ---------------------------------------------------------
# SQLite DB helpers
# ---------------------------------------------------------

def init_db() -> None:
    logger.info("SQLite DB initialized at %s", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id TEXT,
            phone_number TEXT,
            status TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT,
            notes TEXT,
            call_sid TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


init_db()



def _format_ist(ts_str: str) -> str:
    """Convert SQLite timestamp (assumed UTC) to IST string."""
    if not ts_str:
        return ""
    s = str(ts_str).strip()
    # SQLite CURRENT_TIMESTAMP is 'YYYY-MM-DD HH:MM:SS'
    from datetime import datetime, timezone
    dt_obj = None
    try:
        dt_obj = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            # try ISO
            dt_obj = datetime.fromisoformat(s)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        except Exception:
            return s
    try:
        ist = dt_obj.astimezone(ZoneInfo("Asia/Kolkata"))
        return ist.strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return s



# ---------------------------------------------------------
# Audio helpers (24k <-> 8k)
# ---------------------------------------------------------

def downsample_24k_to_8k_pcm16(pcm24: bytes) -> bytes:
    """24 kHz mono PCM16 -> 8 kHz mono PCM16 using stdlib audioop."""
    converted, _ = audioop.ratecv(pcm24, 2, 1, 24000, 8000, None)
    return converted


def upsample_8k_to_24k_pcm16(pcm8: bytes) -> bytes:
    """8 kHz mono PCM16 -> 24 kHz mono PCM16 using stdlib audioop."""
    converted, _ = audioop.ratecv(pcm8, 2, 1, 8000, 24000, None)
    return converted


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


# ---------------------------------------------------------
# PPM Engine helpers (added only; does not change existing flow)
# ---------------------------------------------------------

PPM_ENGINE_URL = (os.getenv("PPM_ENGINE_URL") or "").rstrip("/")
PPM_ENGINE_TIMEOUT = float(os.getenv("PPM_ENGINE_TIMEOUT", "5.0"))


def ppm_engine_enabled() -> bool:
    return bool(PPM_ENGINE_URL)


def ppm_build_voice_context(
    *,
    phone_number: str = "",
    channel: str = "voice",
    segment: str = "cold",
    time_of_day: str = "evening",
    product_type: str = "insurance",
    urgency_level: str = "medium",
    lead_temperature: str = "cold",
    prior_engagement: float = 0.2,
    price_sensitivity: float = 0.5,
    trust_score: float = 0.4,
) -> Dict[str, Any]:
    return {
        "segment": segment,
        "channel": channel,
        "time_of_day": time_of_day,
        "product_type": product_type,
        "urgency_level": urgency_level,
        "lead_temperature": lead_temperature,
        "prior_engagement": prior_engagement,
        "price_sensitivity": price_sensitivity,
        "trust_score": trust_score,
        "phone_number": phone_number,
    }


async def ppm_choose_voice_candidate(context: Dict[str, Any]) -> Dict[str, Any]:
    if not ppm_engine_enabled():
        return {
            "strategy_key": "ppm_disabled",
            "message_text": "",
            "estimated_cost": 0.0,
            "source": "disabled",
        }

    try:
        async with httpx.AsyncClient(timeout=PPM_ENGINE_TIMEOUT) as client:
            resp = await client.post(f"{PPM_ENGINE_URL}/choose-candidate", json=context)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                data.setdefault("source", "ppm")
                return data
    except Exception:
        logger.exception("PPM choose-candidate call failed")

    return {
        "strategy_key": "ppm_error",
        "message_text": "",
        "estimated_cost": 0.0,
        "source": "fallback",
    }


async def ppm_log_voice_outcome(
    *,
    decision_id: Optional[int] = None,
    phone_number: str = "",
    lead_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    segment: str = "cold",
    strategy_key: str = "unknown",
    replied: bool = False,
    converted: bool = False,
    opted_out: bool = False,
    revenue: float = 0.0,
    call_duration_seconds: int = 0,
    outcome_notes: str = "",
) -> Dict[str, Any]:
    if not ppm_engine_enabled():
        return {"ok": False, "reason": "ppm_disabled"}

    payload = {
        "decision_id": decision_id,
        "phone_number": phone_number,
        "lead_id": lead_id,
        "tenant_id": tenant_id,
        "segment": segment,
        "channel": "voice",
        "strategy_key": strategy_key,
        "replied": replied,
        "converted": converted,
        "opted_out": opted_out,
        "revenue": revenue,
        "call_duration_seconds": call_duration_seconds,
        "outcome_notes": outcome_notes,
    }

    try:
        async with httpx.AsyncClient(timeout=PPM_ENGINE_TIMEOUT) as client:
            resp = await client.post(f"{PPM_ENGINE_URL}/log-outcome", json=payload)
            resp.raise_for_status()
            data = resp.json()
            print("PPM OUTCOME RESPONSE:", data)
            return data
    except Exception:
        logger.exception("PPM log-outcome call failed")

    return {"ok": False, "reason": "ppm_outcome_error"}


async def ppm_get_voice_opening_line(
    *,
    phone_number: str = "",
    segment: str = "cold",
    time_of_day: str = "evening",
    product_type: str = "insurance",
    urgency_level: str = "medium",
    lead_temperature: str = "cold",
    prior_engagement: float = 0.2,
    price_sensitivity: float = 0.5,
    trust_score: float = 0.4,
    fallback_text: str = "",
) -> str:
    print("PPM FUNCTION CALLED")
    context = ppm_build_voice_context(
        phone_number=phone_number,
        segment=segment,
        time_of_day=time_of_day,
        product_type=product_type,
        urgency_level=urgency_level,
        lead_temperature=lead_temperature,
        prior_engagement=prior_engagement,
        price_sensitivity=price_sensitivity,
        trust_score=trust_score,
    )
    candidate = await ppm_choose_voice_candidate(context)
    print("PPM RESPONSE:", candidate)
    message_text = (candidate or {}).get("message_text") or ""
    return message_text or fallback_text


def ppm_build_voice_opening_package_local(
    *,
    phone_number: str = "",
    lead_name: str = "",
    industry: str = "insurance",
    product_type: str = "insurance",
    journey_stage: str = "cold",
) -> Dict[str, Any]:
    """
    Additive local voice-opening package.
    Does not remove or rename any existing variable / flow.
    Keeps opening short, human, and authority-based without long monologue.
    """
    safe_lead_name = (lead_name or "").strip()
    safe_industry = (industry or "").strip().lower()
    safe_product_type = (product_type or "").strip().lower()
    safe_journey_stage = (journey_stage or "").strip().lower()

    if safe_lead_name:
        primary_opening = (
            f"Hello {safe_lead_name}... Shashinath bol raha hoon, financial planning side se. "
            "Ek quick yes-no question tha — aapka life cover active hai?"
        )
    elif safe_industry == "insurance" or safe_product_type == "insurance":
        primary_opening = (
            "Hello... Shashinath bol raha hoon, financial planning side se. "
            "Ek quick yes-no question tha — aapka life cover active hai?"
        )
    else:
        primary_opening = (
            "Hello... Shashinath bol raha hoon. "
            "Ek quick yes-no question tha — kya abhi baat karna convenient hai?"
        )

    followup_identity = "Main Shashinath bol raha hoon, financial planning side se."
    followup_context = (
        "Bas isliye pooch raha tha kyunki kaafi families ka cover adequate nahi hota."
    )
    followup_question = "Aapne last time apna cover kab review kiya tha?"
    whatsapp_transition = (
        "Agar aap chaho toh main WhatsApp par ek short coverage check bhej deta hoon."
    )

    interruption_map = {
        "kaun": "Main Shashinath bol raha hoon, financial planning side se. Bas quick insurance check tha.",
        "kaun bol rahe ho": "Main Shashinath bol raha hoon, financial planning side se.",
        "busy": "No problem sir, main WhatsApp par ek short useful message bhej deta hoon.",
        "later": "Theek hai sir, main WhatsApp par ek short useful message bhej deta hoon.",
        "abhi nahi": "Theek hai sir, main WhatsApp par ek short useful message bhej deta hoon.",
        "nahi chahiye": "Bilkul theek sir, force nahi karunga. Main ek short check WhatsApp par bhej deta hoon.",
        "kya hai": "Bas quick insurance coverage check tha, detail mein nahi jaunga.",
        "already policy": "Achha hai sir. Aapne uska last review kab kiya tha?",
    }

    return {
        "primary_opening": primary_opening,
        "followup_identity": followup_identity,
        "followup_context": followup_context,
        "followup_question": followup_question,
        "whatsapp_transition": whatsapp_transition,
        "interruption_map": interruption_map,
        "metadata": {
            "phone_number": phone_number,
            "industry": safe_industry,
            "product_type": safe_product_type,
            "journey_stage": safe_journey_stage,
        },
    }


def ppm_normalize_voice_opening_line_local(
    opening_line: str,
    voice_opening_package: Dict[str, Any],
) -> str:
    """
    Additive normalization only.
    Keeps existing variable names untouched and avoids deleting existing logic.
    """
    normalized_opening_line = (opening_line or "").strip()
    fallback_opening_line = (voice_opening_package or {}).get("primary_opening") or ""

    if not normalized_opening_line:
        return fallback_opening_line

    normalized_lower = normalized_opening_line.lower()

    # If PPM returns a very long ad-like opener, safely compress to the authority+question form.
    if len(normalized_opening_line) > 180:
        return fallback_opening_line

    long_sales_markers = [
        "25 years",
        "mumbai branch",
        "5 mins",
        "5 minutes",
        "underinsured hote hain",
        "just wanted to check",
    ]
    if any(marker in normalized_lower for marker in long_sales_markers):
        return fallback_opening_line

    return normalized_opening_line


@app.get("/ppm/health")
async def ppm_health():
    return {
        "ppm_engine_enabled": ppm_engine_enabled(),
        "ppm_engine_url": PPM_ENGINE_URL,
        "ppm_engine_timeout": PPM_ENGINE_TIMEOUT,
    }


@app.get("/ppm/test-choose")
async def ppm_test_choose():
    context = ppm_build_voice_context()
    candidate = await ppm_choose_voice_candidate(context)
    return {
        "context": context,
        "candidate": candidate,
    }


# ---------------------------------------------------------
# Remote PPM Engine web-service helpers (added only; existing flow unchanged)
# ---------------------------------------------------------

async def ppm_engine_post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not PPM_ENGINE_URL:
        raise RuntimeError("PPM_ENGINE_URL is not set")

    url = f"{PPM_ENGINE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=PPM_ENGINE_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def ppm_log_decision_remote(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await ppm_engine_post("/ppm/log-decision", payload)


async def ppm_log_debug_event_remote(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await ppm_engine_post("/ppm/log-debug-event", payload)


async def ppm_log_outcome_remote(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await ppm_engine_post("/ppm/log-outcome", payload)


@app.get("/ppm/remote-health")
async def ppm_remote_health():
    if not ppm_engine_enabled():
        return {"ok": False, "reason": "ppm_engine_disabled"}
    try:
        async with httpx.AsyncClient(timeout=PPM_ENGINE_TIMEOUT) as client:
            resp = await client.get(f"{PPM_ENGINE_URL}/health")
            return {
                "ok": resp.is_success,
                "status_code": resp.status_code,
                "body": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
            }
    except Exception as e:
        logger.exception("Remote PPM health check failed")
        return {"ok": False, "reason": str(e)}


# In-memory call transcripts keyed by Exotel stream_sid
CALL_TRANSCRIPTS: Dict[str, Dict[str, Any]] = {}

# Prevent multiple bulk batches from running concurrently
BULK_CALL_LOCK = asyncio.Lock()
BULK_CALL_PROGRESS: Dict[str, Dict[str, Any]] = {}

@app.post("/import_call_logs")
async def import_call_logs(request: Request):
    """
    Bulk-import call logs from JSON.
    Expected body:
    {
      "rows": [
        {
          "call_id": "aa05d63a8179...",
          "phone": "08850298070",
          "status": "completed",
          "summary": "Call with 08850298070 (call_id=...).",
          "created_at": "2025-11-27 18:27:39"
        },
        ...
      ]
    }
    """
    body = await request.json()
    rows = body.get("rows", [])
    if not isinstance(rows, list):
        return {"status": "error", "message": "rows must be a list"}

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    inserted = 0
    for r in rows:
        call_id = r.get("call_id") or ""
        phone = r.get("phone") or r.get("phone_number") or ""
        status = r.get("status") or ""
        summary = r.get("summary") or ""
        created_at = r.get("created_at")  # string, we'll trust the value

        # Clean up old placeholders while importing
        placeholder = "Detailed model summary was not available; this is an auto-generated placeholder."
        if placeholder in summary:
            summary = summary.replace(placeholder, "").strip()

        if "customer_phone_number" in summary:
            summary = summary.replace("customer_phone_number", "mobile number")

        if "example_call_id_12345" in summary and call_id:
            summary = summary.replace("example_call_id_12345", str(call_id))

        cur.execute(
            """
            INSERT INTO call_logs (call_id, phone_number, status, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (call_id, phone, status, summary, created_at),
        )
        inserted += 1

    conn.commit()
    conn.close()

    return {"status": "ok", "inserted": inserted}

@app.get("/download-db")
async def download_db():
    """
    Download the SQLite call_logs.db stored on Render persistent disk (/data).
    """
    db_path = "/data/call_logs.db"
    if os.path.exists(db_path):
        return FileResponse(db_path, filename="call_logs.db")
    return {"status": "error", "message": "Database file not found on disk."}



@app.get("/debug-sqlite-call-logs")
async def debug_sqlite_call_logs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, call_id, phone_number, status, summary, created_at "
        "FROM call_logs ORDER BY id DESC LIMIT 20"
    )
    rows = cur.fetchall()
    conn.close()
    return {"rows": rows}


@app.get("/debug/ppm-events")
async def debug_ppm_events(limit: int = 200):
    rows = get_debug_events(limit=limit)
    return {"rows": rows}


# ---------------------------------------------------------
# HTML dashboard page
# ---------------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
  <head>
    <title>Exotel LIC Voicebot Dashboard</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      h1, h2 { color: #333; }
      .section {
        border: 1px solid #ccc;
        padding: 16px;
        margin-bottom: 24px;
        border-radius: 8px;
      }
      label { display: block; margin-bottom: 4px; }
      input[type="text"], input[type="tel"] {
        padding: 6px 8px;
        width: 260px;
        max-width: 90%%;
        margin-bottom: 8px;
      }
      button {
        padding: 8px 12px;
        background: #007bff;
        color: #fff;
        border: none;
        border-radius: 4px;
        cursor: pointer;
      }
      button:disabled {
        background: #999;
        cursor: not-allowed;
      }
      #call-result {
        margin-top: 8px;
        font-family: monospace;
        white-space: pre-wrap;
      }
      table {
        border-collapse: collapse;
        width: 100%%;
        margin-top: 12px;
      }
      table, th, td { border: 1px solid #ccc; }
      th, td { padding: 6px 8px; text-align: left; font-size: 0.9rem; }
    </style>
  </head>
  <body>
    <h1>Exotel LIC Voicebot Backend</h1>

    <div class="section">
      <h2>Single Outbound Call</h2>
      <form id="single-call-form">
        <label for="phone-input">Customer Phone (e.g. 09111717620)</label>
        <input id="phone-input" type="tel" placeholder="Enter phone number" />
        <br />
        <button id="call-button" type="submit">Call Now</button>
      </form>
      <div id="call-result"></div>
    </div>

    <div class="section">
      <h2>Bulk Outbound Call (Sequential)</h2>
      <form id="bulk-call-form" enctype="multipart/form-data">
        <label for="bulk-file-input">Upload Excel (.xlsx/.xls). First column must contain callee numbers.</label>
        <input id="bulk-file-input" type="file" accept=".xlsx,.xls" />
        <br />
        <button id="bulk-call-button" type="submit">Start Bulk Sequential Call</button>
      </form>
      <div id="bulk-call-result"></div>
      <div id="bulk-progress-box" style="margin-top:12px; display:none; border:1px solid #ddd; padding:12px; border-radius:8px;">
        <div><strong>Batch ID:</strong> <span id="bulk-progress-batch-id"></span></div>
        <div><strong>Status:</strong> <span id="bulk-progress-status">idle</span></div>
        <div><strong>Total:</strong> <span id="bulk-progress-total">0</span></div>
        <div><strong>Running:</strong> <span id="bulk-progress-running">0</span></div>
        <div><strong>Completed:</strong> <span id="bulk-progress-completed">0</span></div>
        <div><strong>Failed:</strong> <span id="bulk-progress-failed">0</span></div>
        <div><strong>Current Number:</strong> <span id="bulk-progress-current-number"></span></div>
        <div><strong>Last Message:</strong> <span id="bulk-progress-message"></span></div>
      </div>
    </div>

    <div class="section">
        <h2>Call Logs (Last 50)</h2>
        <button id="refresh-logs">Refresh Logs</button>
        <button id="download-ranked">Download Ranked Leads CSV</button>
        <table id="logs-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Call ID</th>
            <th>Phone</th>
            <th>Status</th>
            <th>Summary</th>
            <th>Created At</th>
          </tr>
        </thead>
        <tbody id="logs-body">
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>MCP Test</h2>
      <p>
        Click the button below to call <code>/test-mcp</code>, which will:
      </p>
      <ul>
        <li>Insert a dummy row into <code>call_logs</code></li>
        <li>Forward a test payload to <code>LIC_CRM_MCP_BASE_URL/test-save</code></li>
      </ul>
      <button id="mcp-test-button">Run MCP Test</button>
      <div id="mcp-result"></div>
    </div>

    <script>
      async function triggerSingleCall(evt) {
        evt.preventDefault();
        const phoneInput = document.getElementById("phone-input");
        const btn = document.getElementById("call-button");
        const resultDiv = document.getElementById("call-result");
        const phone = phoneInput.value.trim();
        if (!phone) {
          resultDiv.textContent = "Please enter a phone number.";
          return;
        }

        btn.disabled = true;
        resultDiv.textContent = "Placing call...";

        try {
          const resp = await fetch("/exotel-outbound-call", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ phone }),
          });
          const data = await resp.json();
          resultDiv.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          resultDiv.textContent = "Error: " + e;
        } finally {
          btn.disabled = false;
        }
      }

      let bulkProgressTimer = null;

      function renderBulkProgress(data) {
        const box = document.getElementById("bulk-progress-box");
        box.style.display = "block";
        document.getElementById("bulk-progress-batch-id").textContent = data.batch_id || "";
        document.getElementById("bulk-progress-status").textContent = data.status || "";
        document.getElementById("bulk-progress-total").textContent = data.total_numbers || 0;
        document.getElementById("bulk-progress-running").textContent = data.running_count || 0;
        document.getElementById("bulk-progress-completed").textContent = data.completed_count || 0;
        document.getElementById("bulk-progress-failed").textContent = data.failed_count || 0;
        document.getElementById("bulk-progress-current-number").textContent = data.current_number || "";
        document.getElementById("bulk-progress-message").textContent = data.message || "";
      }

      async function pollBulkProgress(batchId) {
        try {
          const resp = await fetch(`/bulk-call-progress/${batchId}`);
          const data = await resp.json();
          renderBulkProgress(data);

          if (data.status === "completed" || data.status === "completed_with_errors" || data.status === "failed") {
            if (bulkProgressTimer) {
              clearInterval(bulkProgressTimer);
              bulkProgressTimer = null;
            }
            await loadCallLogs();
          }
        } catch (e) {
          console.error("Bulk progress polling error:", e);
        }
      }

      async function triggerBulkCall(evt) {
        evt.preventDefault();
        const fileInput = document.getElementById("bulk-file-input");
        const btn = document.getElementById("bulk-call-button");
        const resultDiv = document.getElementById("bulk-call-result");

        if (!fileInput.files || !fileInput.files[0]) {
          resultDiv.textContent = "Please choose an Excel file.";
          return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        btn.disabled = true;
        resultDiv.textContent = "Starting bulk sequential call...";

        try {
          const resp = await fetch("/bulk-call-excel-sequential-start", {
            method: "POST",
            body: formData,
          });
          const data = await resp.json();
          resultDiv.textContent = JSON.stringify(data, null, 2);

          if (data.batch_id) {
            renderBulkProgress({
              batch_id: data.batch_id,
              status: "queued",
              total_numbers: data.total_numbers || 0,
              running_count: 0,
              completed_count: 0,
              failed_count: 0,
              current_number: "",
              message: "Batch created. Starting shortly...",
            });

            if (bulkProgressTimer) {
              clearInterval(bulkProgressTimer);
            }
            bulkProgressTimer = setInterval(() => pollBulkProgress(data.batch_id), 2000);
            await pollBulkProgress(data.batch_id);
          }
        } catch (e) {
          resultDiv.textContent = "Error: " + e;
        } finally {
          btn.disabled = false;
        }
      }

      async function loadCallLogs() {
        const tbody = document.getElementById("logs-body");
        tbody.innerHTML = "";
        try {
          const resp = await fetch("/call_logs");
          const data = await resp.json();
          const logs = data.call_logs || [];
          for (const row of logs) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
              <td>${row.id}</td>
              <td>${row.call_id || ""}</td>
              <td>${row.phone_number || ""}</td>
              <td>${row.status || ""}</td>
              <td>${(row.summary || "").slice(0, 1000)}</td>
              <td>${row.created_at_ist || row.created_at || ""}</td>
            `;
            tbody.appendChild(tr);
          }
        } catch (e) {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td colspan="6">Error loading logs: ${e}</td>`;
          tbody.appendChild(tr);
        }
      }

      async function runMcpTest() {
        const btn = document.getElementById("mcp-test-button");
        const div = document.getElementById("mcp-result");
        btn.disabled = true;
        div.textContent = "Calling /test-mcp ...";
        try {
          const resp = await fetch("/test-mcp");
          const data = await resp.json();
          div.textContent = JSON.stringify(data, null, 2);
          await loadCallLogs();
        } catch (e) {
          div.textContent = "Error: " + e;
        } finally {
          btn.disabled = false;
        }
      }
      async function downloadRankedCsv() {
        const btn = document.getElementById("download-ranked");
        btn.disabled = true;
        try {
          // This hits your ML endpoint and the browser will download ranked_customers.csv
          const url = "/ml/ranked-customers.csv?top_k=50";
          window.open(url, "_blank");
        } catch (e) {
          alert("Error starting download: " + e);
        } finally {
          btn.disabled = false;
        }
      }

      document.getElementById("single-call-form")
        .addEventListener("submit", triggerSingleCall);

      document.getElementById("bulk-call-form")
        .addEventListener("submit", triggerBulkCall);

      document.getElementById("refresh-logs")
        .addEventListener("click", loadCallLogs);
      
      document.getElementById("download-ranked")
        .addEventListener("click", downloadRankedCsv);
    /*
      document.getElementById("mcp-test-button")
        .addEventListener("click", runMcpTest);
        */
      // Initial load
      loadCallLogs();
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# ---------------------------------------------------------
# Exotel bootstrap endpoint (for Voicebot applet)
# ---------------------------------------------------------

@app.get("/exotel-ws-bootstrap")
async def exotel_ws_bootstrap():
    """
    Called by Exotel Voicebot applet (Dynamic WebSocket URL).
    Returns the wss:// URL pointing back to this service's /exotel-media route.
    """
    try:
        logger.info(
            "Exotel WS bootstrap called. PUBLIC_BASE_URL=%s, REALTIME_MODEL=%s",
            PUBLIC_BASE_URL,
            REALTIME_MODEL,
        )
        ws_url = public_url("exotel-media").replace("https://", "wss://")
        payload = {"url": ws_url}
        logger.info("Returning Exotel WS URL: %s", payload)
        return JSONResponse(payload)
    except Exception:
        logger.exception("Error in /exotel-ws-bootstrap")
        return JSONResponse({"error": "internal error"}, status_code=500)


# ---------------------------------------------------------
# Simple lead + call log API (JSON, used by dashboard)
# ---------------------------------------------------------

@app.get("/call_logs")
async def get_call_logs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, call_id, phone_number, status, summary, created_at "
        "FROM call_logs ORDER BY id DESC LIMIT 50"
    )
    rows = cur.fetchall()
    conn.close()
    result = [
        {
            "id": r[0],
            "call_id": r[1],
            "phone_number": r[2],
            "status": r[3],
            "summary": r[4],
            "created_at": r[5],
            "created_at_ist": _format_ist(r[5]),
        }
        for r in rows
    ]
    return {"call_logs": result}


@app.post("/lead")
async def create_lead(request: Request):
    data = await request.json()
    name = data.get("name", "").strip()
    phone = data.get("phone", "").strip()
    notes = data.get("notes", "").strip()

    if not phone:
        return JSONResponse({"error": "phone is required"}, status_code=400)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO leads (name, phone, notes, status)
        VALUES (?, ?, ?, ?)
        """,
        (name, phone, notes, "pending"),
    )
    lead_id = cur.lastrowid
    conn.commit()
    conn.close()

    # Trigger Exotel call (single lead)
    result = exotel_outbound_call(phone)
    call_sid = result.get("Call", {}).get("Sid") if isinstance(result, dict) else None

    # Update lead record with call_sid, status
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE leads
        SET call_sid = ?, status = ?
        WHERE id = ?
        """,
        (call_sid, "calling", lead_id),
    )
    conn.commit()
    conn.close()

    return {"lead_id": lead_id, "call_sid": call_sid, "result": result}


@app.get("/leads")
async def get_leads():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, phone, notes, call_sid, status, created_at "
        "FROM leads ORDER BY id DESC LIMIT 100"
    )
    rows = cur.fetchall()
    conn.close()
    result = [
        {
            "id": r[0],
            "name": r[1],
            "phone": r[2],
            "notes": r[3],
            "call_sid": r[4],
            "status": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]
    return {"leads": result}


# ---------------------------------------------------------
# Exotel outbound call trigger
# ---------------------------------------------------------

def exotel_outbound_call(to_number: str) -> Dict[str, Any]:
    """
    Trigger an outbound call using Exotel API, using the same pattern
    that Exotel support gave and which works via curl:

      curl -X POST "https://API_KEY:API_TOKEN@api.exotel.com/v1/Accounts/gouravnxmx1/Calls/connect.json" \\
        -d "From=09111717620" \\
        -d "CallerId=02248904368" \\
        -d "Url=http://my.exotel.com/gouravnxmx1/exoml/start_voice/1077390" \\
        -H "accept: application/json"

    We adapt this to use environment variables:
      - EXOTEL_SID
      - EXOTEL_TOKEN
      - EXO_CALLER_ID
      - EXOTEL_FLOW_URL
    """
    if not EXO_API_KEY or not EXO_API_TOKEN or not EXO_CALLER_ID:
        logger.error("Exotel env missing (EXO_API_KEY / EXO_API_TOKEN / EXO_CALLER_ID); cannot place outbound call.")
        return {"error": "exotel env missing"}

    exotel_url = f"https://{EXO_API_KEY}:{EXO_API_TOKEN}@api.exotel.com/v1/Accounts/{EXO_SID}/Calls/connect.json"
    payload = {
        "From": to_number,          # customer phone (verified) – same as curl "From"
        "CallerId": EXO_CALLER_ID,  # your Exotel number – same as curl "CallerId"
        "Url": EXOTEL_FLOW_URL,     # flow/app URL – same as curl "Url"
    }

    logger.info("Exotel outbound call URL: %s", exotel_url)
    logger.info("Exotel outbound call payload: %s", payload)

    try:
        import requests

        resp = requests.post(exotel_url, data=payload, timeout=15)
        resp.raise_for_status()
        text = resp.text
        logger.info("Exotel outbound call result: %s", text)
        # Exotel returns XML/JSON; we just return the raw text for now
        return {"raw": text}
    except Exception as e:
        logger.exception("Error placing Exotel outbound call: %s", e)
        return {"error": str(e)}


@app.post("/exotel-outbound-call")
async def exotel_outbound_call_endpoint(request: Request):
    """
    Simple HTTP endpoint to trigger an outbound Exotel call from JSON:
      { "phone": "09111717620" }
    Used by the 'Single Outbound Call' form on the dashboard.
    """
    data = await request.json()
    phone = data.get("phone", "").strip()
    if not phone:
        return JSONResponse({"error": "phone is required"}, status_code=400)

    result = exotel_outbound_call(phone)
    return JSONResponse(result)


# ---------------------------------------------------------
# MCP helper: log call summary to LIC CRM MCP DB
# ---------------------------------------------------------

async def log_call_summary_to_db(call_id: str, phone_number: str, summary: str) -> None:
    """
    Calls LIC_CRM_MCP_BASE_URL/test-save with a JSON body for saving call summary
    into the Postgres call_summaries table (via lic_crm_mcp_server.py).

    This is where the REAL call summary (generated by the Realtime model) is
    forwarded to the MCP service.
    """
    if not LIC_CRM_MCP_BASE_URL:
        logger.warning("LIC_CRM_MCP_BASE_URL not set; cannot log summary to MCP DB")
        return

    url = f"{LIC_CRM_MCP_BASE_URL}/test-save"
    payload = {
        "call_id": call_id,
        "phone_number": phone_number,
        "customer_name": "",       # you can fill real name later if you capture it
        "intent": "",
        "interest_score": 0,
        "next_action": "",
        "raw_summary": summary,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            logger.info("Logging REAL call summary to MCP DB: %s %s", url, payload)
            r = await client.post(url, json=payload)
            logger.info("MCP /test-save response: %s %s", r.status_code, r.text)
    except Exception:
        logger.exception("Error logging call summary to MCP server")


# ---------------------------------------------------------
# MCP test endpoint (manual verification)
# ---------------------------------------------------------

@app.get("/test-mcp")
async def test_mcp():
    """
    Manual test endpoint to verify MCP + DB wiring.

    - Inserts a dummy row into call_logs with status = 'test-mcp'
    - Calls LIC_CRM_MCP_BASE_URL/test-save with a dummy payload
    - Returns both the payload and MCP base URL in JSON
    """
    dummy = {
        "call_id": "test-call-123",
        "phone_number": "9999999999",
        "summary": "Test summary from /test-mcp endpoint.",
    }

    # Insert into local SQLite DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO call_logs (call_id, phone_number, status, summary)
        VALUES (?, ?, ?, ?)
        """,
        (dummy["call_id"], dummy["phone_number"], "test-mcp", dummy["summary"]),
    )
    conn.commit()
    conn.close()

    # Forward to MCP (Postgres) using log_call_summary_to_db
    await log_call_summary_to_db(
        dummy["call_id"],
        dummy["phone_number"],
        dummy["summary"],
    )

    return JSONResponse(
        {
            "status": "ok",
            "mcp_base_url": LIC_CRM_MCP_BASE_URL,
            "payload_sent": dummy,
        }
    )
#-----------------------------------------------------
#--------ML end points 
#---------------------------------

# ---------------------------------------------------------
# ML: label calls (supervised training), train model, top 10
# ---------------------------------------------------------

MODEL_PATH = os.getenv("CALL_LOGS_ML_MODEL_PATH", "/data/call_logs_promising_model.joblib")

def get_labeled_data():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Ensure label table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS call_log_labels (
            call_log_id INTEGER PRIMARY KEY,
            purchased INTEGER
        )
    """)

    # Join call_logs + call_log_labels
    cur.execute("""
        SELECT l.call_log_id, c.summary, l.purchased
        FROM call_log_labels l
        JOIN call_logs c ON c.id = l.call_log_id
        ORDER BY l.call_log_id ASC
    """)

    rows = cur.fetchall()
    conn.close()

    texts = []
    labels = []

    for row in rows:
        cid, summary, purchased = row
        if summary and summary.strip():
            texts.append(summary)
            labels.append(int(purchased))

    return texts, labels


@app.post("/label-call-log")
async def label_call_log(request: Request):
    """
    Label a call log as purchased=true/false.
    Body: { "call_log_id": 12, "purchased": true }
    """
    data = await request.json()
    call_log_id = data.get("call_log_id")
    purchased = bool(data.get("purchased", False))

    if not call_log_id:
        return {"status": "error", "message": "call_log_id required"}

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS call_log_labels (
            call_log_id INTEGER PRIMARY KEY,
            purchased INTEGER
        )
    """)

    cur.execute("""
        INSERT OR REPLACE INTO call_log_labels (call_log_id, purchased)
        VALUES (?, ?)
    """, (call_log_id, int(purchased)))

    conn.commit()
    conn.close()

    return {"status": "ok", "call_log_id": call_log_id, "purchased": purchased}


@app.post("/train-ml-call-logs")
async def train_ml_call_logs():
    """
    Train logistic regression based on labeled call logs.
    Saves model to /data disk.
    """
    texts, labels = get_labeled_data()

    if len(texts) < 4:
        return {
            "status": "error",
            "message": f"Need at least 4 labeled rows. Currently have {len(texts)}."
        }

    clf = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(max_iter=200))
    ])

    clf.fit(texts, labels)
    joblib.dump(clf, MODEL_PATH)

    return {
        "status": "ok",
        "message": f"Model trained and saved to {MODEL_PATH}",
        "samples": len(texts)
    }


@app.get("/top10-ml-call-logs")
async def top10_ml_call_logs(limit: int = 10):
    """
    Returns top N promising leads with ML score.
    """
    if not os.path.exists(MODEL_PATH):
        return {
            "status": "error",
            "message": f"Model file not found at {MODEL_PATH}. Train first."
        }

    clf = joblib.load(MODEL_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, call_id, phone_number, summary, created_at
        FROM call_logs
        ORDER BY id DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        cid, call_id, phone, summary, created_at = r
        if not summary:
            continue
        score = clf.predict_proba([summary])[0][1]  # probability of purchase
        results.append({
            "call_log_id": cid,
            "call_id": call_id,
            "phone": phone,
            "summary": summary,
            "ml_score": float(score),
            "created_at": created_at
        })

    results = sorted(results, key=lambda x: x["ml_score"], reverse=True)
    return {"status": "ok", "top": results[:limit]}

@app.get("/ml/ranked-customers.csv")
async def download_ranked_customers_csv(top_k: int = Query(50, ge=1, le=500)):
    """
    Generate ML-based ranking of customers and return as CSV download.
    """
    df = _load_call_logs_df()
    if df.empty:
        # empty CSV
        csv_bytes = b""
    else:
        df_scored = _build_interest_scores(df)
        top_customers = _rank_customers(df_scored, top_k)
        csv_bytes = top_customers.to_csv(index=False).encode("utf-8")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="ranked_customers.csv"'
        },
    )


#------call_logs backup endpoint------
#----------------------------------
@app.get("/backup/call_logs")
async def backup_call_logs():
    """
    Backup endpoint: returns all call_logs and call_log_labels as JSON.

    Use this to take a snapshot of your data periodically.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) Dump call_logs
    cur.execute(
        """
        SELECT id, call_id, phone_number, status, summary, created_at
        FROM call_logs
        ORDER BY id ASC
        """
    )
    call_logs_rows = cur.fetchall()

    call_logs = [
        {
            "id": r[0],
            "call_id": r[1],
            "phone_number": r[2],
            "status": r[3],
            "summary": r[4],
            "created_at": r[5],
        }
        for r in call_logs_rows
    ]

    # 2) Dump call_log_labels (if table exists)
    labels = []
    try:
        cur.execute(
            """
            SELECT call_log_id, purchased
            FROM call_log_labels
            ORDER BY call_log_id ASC
            """
        )
        label_rows = cur.fetchall()
        labels = [
            {
                "call_log_id": r[0],
                "purchased": bool(r[1]),
            }
            for r in label_rows
        ]
    except sqlite3.OperationalError:
        # Table might not exist on older deployments; don't crash backup
        labels = []

    conn.close()

    return {
        "status": "ok",
        "call_logs_count": len(call_logs),
        "labels_count": len(labels),
        "call_logs": call_logs,
        "call_log_labels": labels,
    }


# ---------------------------------------------------------
# Exotel <-> OpenAI Realtime WebSocket bridge
# ---------------------------------------------------------

@app.websocket("/exotel-media")
async def exotel_media(ws: WebSocket):
    """
    Bi-directional WS:
     - Exotel sends Twilio-style events (connected/start/media/stop).
     - We connect to OpenAI Realtime and stream audio in/out.
    """
    await ws.accept()
    logger.info("Exotel WebSocket connected")

    # Call metadata (per stream)
    call_id: Optional[str] = None
    caller_number: Optional[str] = None
    stream_sid: Optional[str] = None
    call_start_ts: Optional[float] = None
    had_audio: bool = False
    first_media_logged: bool = False
    # =========================
# ADDITIVE VOICE STATE CONTROL
# =========================
    auto_speak_task: Optional[asyncio.Task] = None
    reengage_task: Optional[asyncio.Task] = None
    caller_spoke_flag: bool = False
    # NEW: transcript capture buffers (Option B)
    ai_transcript_texts: list[str] = []
    summary_saved: bool = False  # tracks if model already saved summary


    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY; closing Exotel stream.")
        await ws.close()
        return

    # Exotel stream sequence/timing
    seq_num = 1
    chunk_num = 1
    start_ts = time.time()

    # OpenAI Realtime session
    openai_session: Optional[ClientSession] = None
    openai_ws = None
    pump_task: Optional[asyncio.Task] = None

    async def send_openai(payload: dict):
        """Send JSON payload to OpenAI Realtime WS."""
        nonlocal openai_ws
        if not openai_ws or openai_ws.closed:
            logger.warning("Cannot send to OpenAI: WS not ready")
            return
        t = payload.get("type")
        logger.debug("→ OpenAI: %s", t)
        await openai_ws.send_json(payload)

    async def send_audio_to_exotel(pcm8: bytes):
        """
        Send 8 kHz mono PCM16 back to Exotel as base64 "media" frames.
        Exotel expects 20 ms = 160 samples => 320 bytes per frame.
        """
        nonlocal seq_num, chunk_num, start_ts, stream_sid

        if not stream_sid:
            logger.warning("No stream_sid; cannot send audio to Exotel yet")
            return

        FRAME_BYTES = 320  # 20 ms at 8kHz mono 16-bit
        now_ms = lambda: int((time.time() - start_ts) * 1000)

        for i in range(0, len(pcm8), FRAME_BYTES):
            chunk_bytes = pcm8[i: i + FRAME_BYTES]
            if not chunk_bytes:
                continue

            payload_b64 = base64.b64encode(chunk_bytes).decode("ascii")
            ts = now_ms()

            msg = {
                "event": "media",
                "stream_sid": stream_sid,
                "sequence_number": str(seq_num),
                "media": {
                    "chunk": str(chunk_num),
                    "timestamp": str(ts),
                    "payload": payload_b64,
                },
            }

            await ws.send_text(json.dumps(msg))
            logger.debug(
                "Sent audio media to Exotel (seq=%s, chunk=%s, bytes=%s)",
                seq_num,
                chunk_num,
                len(chunk_bytes),
            )

            seq_num += 1
            chunk_num += 1

    async def connect_openai(conn_call_id: str, conn_caller_number: str):
        """
        Connect to OpenAI Realtime, configure LIC persona + tools,
        and start the pump() loop that sends audio back to Exotel and
        handles MCP-style tool-calls.
        """
        nonlocal openai_session, openai_ws, pump_task

        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            }

            url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

            openai_session = ClientSession()
            logger.info("Connecting to OpenAI Realtime WS...")
            openai_ws = await openai_session.ws_connect(url, headers=headers)
            logger.info("OpenAI Realtime WS connected.")
            save_ppm_debug_event(
                call_id=conn_call_id or "",
                stream_sid=stream_sid or "",
                phone_number=conn_caller_number or "",
                event_type="openai_connected",
                level="INFO",
                source="voicebot",
                message="OpenAI Realtime websocket connected",
                payload={"model": REALTIME_MODEL},
            )

            try:
                await ppm_log_debug_event_remote({
                    "client_id": "lic_shashinath",
                    "channel": "voice",
                    "call_id": conn_call_id or "",
                    "stream_sid": stream_sid or "",
                    "phone_number": conn_caller_number or "",
                    "event_type": "openai_connected",
                    "level": "INFO",
                    "message": "OpenAI realtime websocket connected",
                    "payload": {"model": REALTIME_MODEL},
                })
            except Exception:
                logger.exception("Remote PPM log-debug-event failed (openai_connected)")

            # Build instructions for LIC agent persona
            instructions_text = (
                "You are Mr. Shashinath Thakur, a highly experienced Financial planner with 25 years of experience.\n\n"

                "VOICE & TONE:\n"
                "- Speak in natural Hinglish (Mumbai style).\n"
                "- Voice should be calm, confident, slightly deep and manly.\n"
                "- Never use words like 'bhaiya'. Use 'sir' or neutral tone.\n"
                "- Speak slowly, with authority and clarity.\n"
                "- Use very short sentences (8–10 words max).\n"
                "- Pause frequently and let the customer speak.\n\n"

                "PRIMARY GOAL:\n"
                "- Do NOT sell immediately.\n"
                "- Trigger curiosity and make the customer think.\n"
                "- Start a natural conversation, not a sales pitch.\n\n"

                "OPENING RULE (VERY IMPORTANT):\n"
                "- The FIRST spoken line of the call is controlled by PPM Engine (Profit Per Message).\n"
                "- The first spoken line MUST be exactly the PPM opening line provided separately.\n"
                "- Do NOT add any greeting, introduction, city, or extra words before that first line.\n"
                "- Do NOT say 'calling from Mumbai' unless the customer asks or the PPM line itself says so.\n"
                "- If useful after the customer responds, you may briefly introduce yourself as Mr. Shashinath Thakur.\n"
                "- After the first line, STOP speaking and wait.\n\n"

                "CONVERSATION FLOW:\n"
                "1. If customer responds → acknowledge briefly.\n"
                "2. Ask ONLY ONE simple follow-up.\n"
                "   Example: 'Aapne last kab apni policy review ki thi?'\n"
                "3. Keep conversation light and natural.\n"
                "4. Do NOT explain products unless asked.\n\n"

                "OBJECTION HANDLING:\n"
                "- If 'busy': 'No problem sir, main WhatsApp pe ek useful cheez share kar deta hoon.'\n"
                "- If 'not interested': 'Totally fine sir, bas ek small insight bhej deta hoon WhatsApp pe.'\n"
                "- If 'already have policy': 'Great sir, most people still have gaps… ek quick check kar sakte hain.'\n\n"

                "WHATSAPP TRANSITION:\n"
                "- Try to move conversation to WhatsApp.\n"
                "- Example: 'Should I share a quick calculation on WhatsApp?'\n\n"

                "DO NOT:\n"
                "- Sound like a salesman\n"
                "- Speak long paragraphs\n"
                "- Push aggressively\n"
                "- Invent your own opening line when a PPM opening line is provided\n\n"

                "END GOAL:\n"
                "- Get engagement OR WhatsApp permission\n\n"

                "VERY IMPORTANT TOOL RULE:\n"
                "- After the conversation is finished, call 'save_call_summary' exactly once.\n"
                "- Include call_id, phone_number, and summary (interest level, objection, next step).\n"
            )
            tools_spec = [
                {
                    "type": "function",
                    "name": "save_call_summary",
                    "description": "Persist a structured Financial Planning call summary into the CRM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "call_id": {
                                "type": "string",
                                "description": "Unique call id (Exotel CallSid or generated).",
                            },
                            "phone_number": {
                                "type": "string",
                                "description": "Customer phone number with country code.",
                            },
                            "summary": {
                                "type": "string",
                                "description": (
                                    "Short structured summary of the call, including "
                                    "customer needs, recommended plans, and next steps."
                                ),
                            },
                        },
                        "required": ["call_id", "phone_number", "summary"],
                    },
                }
            ]

            session_config: dict = {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "alloy",
                "turn_detection": {"type": "server_vad"},
                "instructions": instructions_text,
                "tools": tools_spec,
            }

            # Send initial session.update
            await send_openai({"type": "session.update", "session": session_config})
            logger.info("Sent session.update with LIC persona + tools config")
            save_ppm_debug_event(
                call_id=conn_call_id or "",
                stream_sid=stream_sid or "",
                phone_number=conn_caller_number or "",
                event_type="openai_session_update",
                level="INFO",
                source="voicebot",
                message="session.update sent to OpenAI",
            )

            try:
                await ppm_log_debug_event_remote({
                    "client_id": "lic_shashinath",
                    "channel": "voice",
                    "call_id": conn_call_id or "",
                    "stream_sid": stream_sid or "",
                    "phone_number": conn_caller_number or "",
                    "event_type": "openai_session_update",
                    "level": "INFO",
                    "message": "session.update sent to OpenAI",
                    "payload": {},
                })
            except Exception:
                logger.exception("Remote PPM log-debug-event failed (openai_session_update)")

            ppm_context = ppm_build_voice_context(
                phone_number=conn_caller_number or "",
                segment="cold",
                time_of_day="evening",
                product_type="insurance",
                urgency_level="medium",
                lead_temperature="cold",
                prior_engagement=0.2,
                price_sensitivity=0.5,
                trust_score=0.4,
            )

            ppm_candidate = await ppm_choose_voice_candidate(ppm_context)

            print("PPM RESPONSE:", ppm_candidate)

            ppm_opening_line = (ppm_candidate or {}).get("message_text")

            save_ppm_decision(
                call_id=conn_call_id or "",
                phone_number=conn_caller_number or "",
                decision_id=str((ppm_candidate or {}).get("decision_id") or ""),
                strategy_key=(ppm_candidate or {}).get("strategy_key", "") or "",
                selected_message=(ppm_candidate or {}).get("message_text", "") or "",
                predicted_conversion=float((ppm_candidate or {}).get("pred_conv") or 0.0),
                predicted_optout=float((ppm_candidate or {}).get("pred_optout") or 0.0),
                expected_value=float((ppm_candidate or {}).get("expected_value") or 0.0),
                source=(ppm_candidate or {}).get("source", "voice") or "voice",
            )

            save_ppm_debug_event(
                call_id=conn_call_id or "",
                stream_sid=stream_sid or "",
                phone_number=conn_caller_number or "",
                event_type="ppm_candidate_selected",
                level="INFO",
                source="ppm",
                message="PPM candidate selected in connect_openai",
                payload={
                    "ppm_candidate": ppm_candidate,
                    "ppm_context": ppm_context,
                },
            )

            try:
                await ppm_log_decision_remote({
                    "client_id": "lic_shashinath",
                    "channel": "voice",
                    "journey_id": "voice_opening",
                    "call_id": conn_call_id or "",
                    "phone_number": conn_caller_number or "",
                    "decision_id": str((ppm_candidate or {}).get("decision_id") or ""),
                    "strategy_key": (ppm_candidate or {}).get("strategy_key", "") or "",
                    "message_text": (ppm_candidate or {}).get("message_text", "") or "",
                    "predicted_conversion": float((ppm_candidate or {}).get("pred_conv") or 0.0),
                    "predicted_optout": float((ppm_candidate or {}).get("pred_optout") or 0.0),
                    "expected_value": float((ppm_candidate or {}).get("expected_value") or 0.0),
                    "context": ppm_context,
                })
            except Exception:
                logger.exception("Remote PPM log-decision failed")

            if not ppm_opening_line:
                print("⚠️ PPM FAILED — using fallback (should be rare)")

                ppm_opening_line = (
                    "Sir ek second — agar main galat hoon toh aap turant cut kar dena,"
                    "bas ek quick check karna tha — aapka insurance recently review hua hai kya?"
                )

            voice_opening_package = ppm_build_voice_opening_package_local(
                phone_number=conn_caller_number or "",
                lead_name="",
                industry="insurance",
                product_type="insurance",
                journey_stage="cold",
            )

            ppm_opening_line_before_local_normalization = ppm_opening_line
            ppm_opening_line = ppm_normalize_voice_opening_line_local(
                ppm_opening_line,
                voice_opening_package,
            )

            ppm_followup_identity = (voice_opening_package or {}).get(
                "followup_identity",
                "Main Shashinath bol raha hoon, financial planning side se.",
            )
            ppm_followup_context = (voice_opening_package or {}).get(
                "followup_context",
                "Bas isliye pooch raha tha kyunki kaafi families ka cover adequate nahi hota.",
            )
            ppm_followup_question = (voice_opening_package or {}).get(
                "followup_question",
                "Aapne last time apna cover kab review kiya tha?",
            )
            ppm_whatsapp_transition = (voice_opening_package or {}).get(
                "whatsapp_transition",
                "Agar aap chaho toh main WhatsApp par ek short coverage check bhej deta hoon.",
            )

            save_ppm_debug_event(
                call_id=conn_call_id or "",
                stream_sid=stream_sid or "",
                phone_number=conn_caller_number or "",
                event_type="ppm_opening_line_normalized_local",
                level="INFO",
                source="ppm",
                message="Local additive normalization applied to opening line",
                payload={
                    "opening_line_before_local_normalization": ppm_opening_line_before_local_normalization,
                    "opening_line_after_local_normalization": ppm_opening_line,
                    "voice_opening_package": voice_opening_package,
                },
            )

            try:
                await ppm_log_debug_event_remote({
                    "client_id": "lic_shashinath",
                    "channel": "voice",
                    "call_id": conn_call_id or "",
                    "stream_sid": stream_sid or "",
                    "phone_number": conn_caller_number or "",
                    "event_type": "ppm_opening_line_normalized_local",
                    "level": "INFO",
                    "message": "Local additive normalization applied to opening line",
                    "payload": {
                        "opening_line_before_local_normalization": ppm_opening_line_before_local_normalization,
                        "opening_line_after_local_normalization": ppm_opening_line,
                        "voice_opening_package": voice_opening_package,
                    },
                })
            except Exception:
                logger.exception("Remote PPM log-debug-event failed (ppm_opening_line_normalized_local)")

            save_ppm_debug_event(
                call_id=conn_call_id or "",
                stream_sid=stream_sid or "",
                phone_number=conn_caller_number or "",
                event_type="ppm_opening_line_final",
                level="INFO",
                source="ppm",
                message="Final opening line selected for call",
                payload={
                    "opening_line": ppm_opening_line,
                    "used_fallback": not bool((ppm_candidate or {}).get("message_text")),
                    "decision_id": (ppm_candidate or {}).get("decision_id"),
                    "strategy_key": (ppm_candidate or {}).get("strategy_key"),
                },
            )

            # store decision metadata for this call
            if stream_sid and stream_sid in CALL_TRANSCRIPTS:
                CALL_TRANSCRIPTS[stream_sid]["ppm_decision_id"] = (ppm_candidate or {}).get("decision_id")
                CALL_TRANSCRIPTS[stream_sid]["ppm_strategy_key"] = (ppm_candidate or {}).get("strategy_key", "unknown")
                CALL_TRANSCRIPTS[stream_sid]["ppm_source"] = (ppm_candidate or {}).get("source", "")

                CALL_TRANSCRIPTS[stream_sid]["ppm_opening_line"] = ppm_opening_line
                CALL_TRANSCRIPTS[stream_sid]["ppm_followup_identity"] = ppm_followup_identity
                CALL_TRANSCRIPTS[stream_sid]["ppm_followup_context"] = ppm_followup_context
                CALL_TRANSCRIPTS[stream_sid]["ppm_followup_question"] = ppm_followup_question
                CALL_TRANSCRIPTS[stream_sid]["ppm_whatsapp_transition"] = ppm_whatsapp_transition


            instructions_text = instructions_text + (
                "\n\n"
                "LOCAL ADDITIVE OPENING RULES:\n"
                f"- Preferred opener for this call: '{ppm_opening_line}'\n"
                f"- If the customer responds, identity line can be: '{ppm_followup_identity}'\n"
                f"- Then use short context line: '{ppm_followup_context}'\n"
                f"- Then use one follow-up question: '{ppm_followup_question}'\n"
                f"- If customer is busy, use WhatsApp transition: '{ppm_whatsapp_transition}'\n"
                "- Do not give long statistic-led monologues in the opening turn.\n"
                "- Keep the opener authority-based, human, and short."
            )

            await send_openai({"type": "session.update", "session": {"instructions": instructions_text}})

            save_ppm_debug_event(
                call_id=conn_call_id or "",
                stream_sid=stream_sid or "",
                phone_number=conn_caller_number or "",
                event_type="openai_session_update_opening_rules_added",
                level="INFO",
                source="voicebot",
                message="Additional opening rules appended via additive session.update",
                payload={
                    "ppm_opening_line": ppm_opening_line,
                    "ppm_followup_identity": ppm_followup_identity,
                    "ppm_followup_context": ppm_followup_context,
                    "ppm_followup_question": ppm_followup_question,
                    "ppm_whatsapp_transition": ppm_whatsapp_transition,
                },
            )

            try:
                await ppm_log_debug_event_remote({
                    "client_id": "lic_shashinath",
                    "channel": "voice",
                    "call_id": conn_call_id or "",
                    "stream_sid": stream_sid or "",
                    "phone_number": conn_caller_number or "",
                    "event_type": "openai_session_update_opening_rules_added",
                    "level": "INFO",
                    "message": "Additional opening rules appended via additive session.update",
                    "payload": {
                        "ppm_opening_line": ppm_opening_line,
                        "ppm_followup_identity": ppm_followup_identity,
                        "ppm_followup_context": ppm_followup_context,
                        "ppm_followup_question": ppm_followup_question,
                        "ppm_whatsapp_transition": ppm_whatsapp_transition,
                    },
                })
            except Exception:
                logger.exception("Remote PPM log-debug-event failed (openai_session_update_opening_rules_added)")

            # Ask the model to start the first greeting turn
            await send_openai(
                {
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "Start the call now.\n"
                            "Speak in Hinglish, calm and confident tone.\n"
                            "Say exactly the PPM opening line given below.\n"
                            "Do not add any greeting, city, name, introduction, or extra words before it.\n"
                            "Do not paraphrase it.\n"
                            "Finish within 10 seconds.\n"
                            "Then STOP and listen.\n\n"
                            f"Say exactly this and nothing before it:\n'{ppm_opening_line}'\n\n"
                            "Pause after speaking."
                        ),
                        "modalities": ["text", "audio"],
                      },
                }
            )
            # =========================
                # ADDITIVE AUTO-SPEAK + RE-ENGAGE (DO NOT MODIFY EXISTING FLOW)
            # =========================

            if VOICE_ENABLE_START_NUDGE:

                async def _auto_speak_nudge():
                    await asyncio.sleep(VOICE_AUTO_SPEAK_DELAY_MS / 1000.0)

                    if not caller_spoke_flag:
                        try:
                            await send_openai({
                                "type": "response.create",
                                "response": {
                                    "instructions": "Continue speaking the opening line naturally if user is silent.",
                                    "modalities": ["text", "audio"],
                                },
                            })
                        except Exception:
                            logger.exception("Auto speak nudge failed")

                    async def _reengage_nudge():
                        await asyncio.sleep(VOICE_REENGAGE_DELAY_MS / 1000.0)

                        if not caller_spoke_flag:
                            try:
                                await send_openai({
                                    "type": "response.create",
                                    "response": {
                                        "instructions": (
                                            "User seems silent. Say: "
                                            "'Aap sun pa rahe hain? Ek quick point bolke nikal jaunga.'"
                                        ),
                                        "modalities": ["text", "audio"],
                                    },
                                })
                            except Exception:
                                logger.exception("Re-engage nudge failed")

                            auto_speak_task = asyncio.create_task(_auto_speak_nudge())
                            reengage_task = asyncio.create_task(_reengage_nudge())

            async def pump():
                """
                Receive events from OpenAI Realtime and:
                  - forward audio deltas to Exotel
                  - capture tool calls and forward them to MCP HTTP endpoint
                """
                try:
                    async for msg in openai_ws:
                        if msg.type != WSMsgType.TEXT:
                            continue

                        try:
                            evt = json.loads(msg.data)
                        except Exception:
                            logger.exception("Failed to parse OpenAI WS message")
                            continue

                        et = evt.get("type")
                        logger.debug("OpenAI EVENT: %s - %s", et, evt)

                        if et == "response.output_text.delta":
                            text_chunk = evt.get("text") or ""
                            if text_chunk:
                                ai_transcript_texts.append(text_chunk)

                        # Audio deltas from model
                        if et in ("response.audio.delta", "response.output_audio.delta"):
                            delta = evt.get("delta") or evt.get("audio") or {}
                            # In newer Realtime responses, `delta` may be either:
                            #   - a dict: { "audio": "<base64>" } or { "data": "<base64>" }
                            #   - a raw base64 string
                            if isinstance(delta, str):
                                b64 = delta
                            else:
                                b64 = delta.get("audio") or delta.get("data")
                            if not b64:
                                continue

                            try:
                                pcm24 = base64.b64decode(b64)
                            except Exception:
                                logger.exception("Failed to decode audio delta")
                                continue

                            try:
                                pcm8 = downsample_24k_to_8k_pcm16(pcm24)
                            except Exception:
                                logger.exception("Downsampling 24k -> 8k failed")
                                continue

                            await send_audio_to_exotel(pcm8)

                        # TOOL CALL: where call summary is GENERATED and logged
                        elif et == "response.function_call_arguments.done":
                            name = evt.get("name")
                            arg_str = evt.get("arguments") or "{}"

                            try:
                                args = json.loads(arg_str)
                            except Exception:
                                logger.exception(
                                    "Failed to parse tool arguments JSON: %r", arg_str
                                )
                                continue

                            logger.info("Tool-call done: name=%s args=%s", name, args)

                            if name == "save_call_summary":
                                call_id_param = args.get("call_id") or conn_call_id
                                phone_param = args.get("phone_number") or conn_caller_number
                                summary_param = (args.get("summary") or "").strip()
                                summary_saved = True


                                # Append total call duration if we know it
                                duration_seconds = None
                                if call_start_ts:
                                    try:
                                        duration_seconds = int(time.time() - call_start_ts)
                                    except Exception:
                                        duration_seconds = None

                                if duration_seconds is not None:
                                    m, s = divmod(duration_seconds, 60)
                                    duration_text = f"{m}m {s}s" if m > 0 else f"{s}s"
                                    # Avoid double-appending if model already mentioned it
                                    if "Total call duration:" not in summary_param:
                                        if summary_param:
                                            summary_param = (
                                                summary_param
                                                + f"\n\nTotal call duration: {duration_text}."
                                            )
                                        else:
                                            summary_param = (
                                                f"Caller did not speak anything during the call.\n\n"
                                                f"Total call duration: {duration_text}."
                                            )

                                logger.info(
                                    "REAL SUMMARY RECEIVED FROM MODEL (with duration): %s",
                                    summary_param,
                                )

                                # Save REAL summary into local SQLite DB
                                conn = sqlite3.connect(DB_PATH)
                                cur = conn.cursor()
                                cur.execute(
                                    """
                                    INSERT INTO call_logs (call_id, phone_number, status, summary)
                                    VALUES (?, ?, ?, ?)
                                    """,
                                    (
                                        call_id_param,
                                        phone_param,
                                        "completed",
                                        summary_param,
                                    ),
                                )
                                conn.commit()
                                conn.close()

                                save_ppm_outcome(
                                    call_id=call_id_param or "",
                                    phone_number=phone_param or "",
                                    outcome_status="completed",
                                    reward_value=0.0,
                                    summary=summary_param,
                                    duration_seconds=float(duration_seconds or 0),
                                )

                                save_ppm_debug_event(
                                    call_id=call_id_param or "",
                                    stream_sid=stream_sid or "",
                                    phone_number=phone_param or "",
                                    event_type="call_summary_saved",
                                    level="INFO",
                                    source="voicebot",
                                    message="Model-generated call summary saved",
                                    payload={
                                        "status": "completed",
                                        "summary": summary_param,
                                        "duration_seconds": duration_seconds,
                                    },
                                )


                                try:
                                    await ppm_log_debug_event_remote({
                                        "client_id": "lic_shashinath",
                                        "channel": "voice",
                                        "call_id": call_id_param or "",
                                        "stream_sid": stream_sid or "",
                                        "phone_number": phone_param or "",
                                        "event_type": "call_summary_saved",
                                        "level": "INFO",
                                        "message": "Model-generated call summary saved",
                                        "payload": {
                                            "status": "completed",
                                            "summary": summary_param,
                                            "duration_seconds": duration_seconds,
                                        },
                                    })
                                except Exception:
                                    logger.exception("Remote PPM log-debug-event failed (call_summary_saved)")

                                try:
                                    await ppm_log_outcome_remote({
                                        "client_id": "lic_shashinath",
                                        "channel": "voice",
                                        "call_id": call_id_param or "",
                                        "phone_number": phone_param or "",
                                        "decision_id": str((ppm_candidate or {}).get("decision_id") or ""),
                                        "outcome_status": "completed",
                                        "reward_value": 0.0,
                                        "summary": summary_param,
                                        "duration_seconds": float(duration_seconds or 0),
                                    })
                                except Exception:
                                    logger.exception("Remote PPM log-outcome failed (call_summary_saved)")

                                # Forward REAL summary to MCP Postgres DB
                                await log_call_summary_to_db(
                                    call_id_param,
                                    phone_param,
                                    summary_param,
                                )

                                # Let the model know tool-call succeeded
                                await send_openai(
                                    {
                                        "type": "response.create",
                                        "response": {
                                            "instructions": (
                                                "I have saved the call summary to the CRM. "
                                                "Thank the customer politely and end the call."
                                            ),
                                            "modalities": ["text", "audio"],
                                        },
                                    }
                                )


                                # Forward REAL summary to MCP Postgres DB
                                await log_call_summary_to_db(
                                    call_id_param,
                                    phone_param,
                                    summary_param,
                                )

                                # Let the model know tool-call succeeded
                                await send_openai(
                                    {
                                        "type": "response.create",
                                        "response": {
                                            "instructions": (
                                                "I have saved the call summary to the CRM. "
                                                "Thank the customer politely and end the call."
                                            ),
                                            "modalities": ["text", "audio"],
                                        },
                                    }
                                )

                        elif et in (
                            "response.audio.done",
                            "response.output_audio.done",
                            "response.done",
                        ):
                            logger.info("OpenAI response finished.")

                        elif et == "error":
                            logger.error("OpenAI ERROR event: %s", evt)
                            save_ppm_debug_event(
                                call_id=conn_call_id or "",
                                stream_sid=stream_sid or "",
                                phone_number=conn_caller_number or "",
                                event_type="openai_error_event",
                                level="ERROR",
                                source="voicebot",
                                message="OpenAI error event received",
                                payload=evt,
                            )

                except Exception as e:
                    logger.exception("Pump error: %s", e)
                    save_ppm_debug_event(
                        call_id=conn_call_id or "",
                        stream_sid=stream_sid or "",
                        phone_number=conn_caller_number or "",
                        event_type="pump_error",
                        level="ERROR",
                        source="voicebot",
                        message=str(e),
                    )

            pump_task = asyncio.create_task(pump())

        except Exception as e:
            logger.exception("OpenAI connection error: %s", e)
            save_ppm_debug_event(
                call_id=conn_call_id or "",
                stream_sid=stream_sid or "",
                phone_number=conn_caller_number or "",
                event_type="openai_connection_error",
                level="ERROR",
                source="voicebot",
                message=str(e),
            )

    try:
        openai_started = False

        while True:
            raw = await ws.receive_text()
            evt = json.loads(raw)
            ev = evt.get("event")
            logger.info("Exotel EVENT: %s - msg=%s", ev, evt)

            if ev == "connected":
                # initial handshake from Exotel
                continue

            elif ev == "start":
				
                logger.info("Exotel sent Start event --GV ")
                start_obj = evt.get("start") or {}
                stream_sid = start_obj.get("stream_sid") or evt.get("stream_sid")
                start_ts = time.time()
                call_start_ts = time.time()

                call_id = start_obj.get("call_sid") or start_obj.get("callSid") or evt.get("call_sid")
                caller_number = (
                    start_obj.get("from")
                    or start_obj.get("caller_id")
                    or start_obj.get("caller_number")
                    or ""
                )

                logger.info(
                    "Exotel start: stream_sid=%s call_id=%s caller=%s",
                    stream_sid,
                    call_id,
                    caller_number,
                )

                CALL_TRANSCRIPTS[stream_sid] = {
                    "call_id": call_id,
                    "phone_number": caller_number,
                    "turns": [],  # list of (speaker, text)
                }

                save_ppm_debug_event(
                    call_id=call_id or "",
                    stream_sid=stream_sid or "",
                    phone_number=caller_number or "",
                    event_type="exotel_start",
                    level="INFO",
                    source="voicebot",
                    message="Exotel start event received",
                    payload=evt,
                )

                try:
                    await ppm_log_debug_event_remote({
                        "client_id": "lic_shashinath",
                        "channel": "voice",
                        "call_id": call_id or "",
                        "stream_sid": stream_sid or "",
                        "phone_number": caller_number or "",
                        "event_type": "exotel_start",
                        "level": "INFO",
                        "message": "Exotel start event received",
                        "payload": evt,
                    })
                except Exception:
                    logger.exception("Remote PPM log-debug-event failed (exotel_start)")

                try:
                    await log_call_summary_to_db(
                        call_id or stream_sid or "unknown_call",
                        caller_number or "",
                        f"Debug insert from ws_server.py start event for call_id={call_id}, phone={caller_number}",
                    )
                    logger.info("Debug MCP insert from start event completed")
                except Exception:
                    logger.exception("Debug MCP insert from start event FAILED")


                if not openai_started:
                    openai_started = True
                    await connect_openai(call_id or "unknown_call", caller_number or "")

            elif ev == "media":
                # Caller audio (8kHz PCM16) -> upsample to 24kHz -> send to OpenAI
				
                logger.info("Exotel sent Media event --GV ")
                media = evt.get("media") or {}
                payload_b64 = media.get("payload")
                if payload_b64 and openai_ws and not openai_ws.closed:
                    try:
                        pcm8 = base64.b64decode(payload_b64)
                        had_audio = True   # NEW
                        if not first_media_logged:
                            first_media_logged = True
                            save_ppm_debug_event(
                                call_id=call_id or "",
                                stream_sid=stream_sid or "",
                                phone_number=caller_number or "",
                                event_type="exotel_first_media",
                                level="INFO",
                                source="voicebot",
                                message="First caller audio frame received",
                                payload={"payload_bytes": len(pcm8)},
                            )

                            try:
                                await ppm_log_debug_event_remote({
                                    "client_id": "lic_shashinath",
                                    "channel": "voice",
                                    "call_id": call_id or "",
                                    "stream_sid": stream_sid or "",
                                    "phone_number": caller_number or "",
                                    "event_type": "exotel_first_media",
                                    "level": "INFO",
                                    "message": "First caller audio frame received",
                                    "payload": {"payload_bytes": len(pcm8)},
                                })
                            except Exception:
                                logger.exception("Remote PPM log-debug-event failed (exotel_first_media)")
                    except Exception:
                        logger.warning("Invalid base64 in Exotel media payload")
                        continue
                    
                    had_audio = True

                    caller_spoke_flag = True

                    # cancel nudges once user speaks
                    if auto_speak_task:
                        auto_speak_task.cancel()
                    if reengage_task:
                        reengage_task.cancel()

                    pcm24 = upsample_8k_to_24k_pcm16(pcm8)
                    audio_b64 = base64.b64encode(pcm24).decode("ascii")
                    await send_openai(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        }
                    )
                    # NOTE: With server_vad we do NOT call input_audio_buffer.commit manually.
                    # The server will commit automatically when it detects end-of-speech.

            elif ev == "stop":
                logger.info("Exotel sent stop; closing WS and letting model wrap up.")

                #--------------------------------------added for improved summary on stop
                # --- NEW: Duration and fallback summary logic (Option B) ---
                # Compute duration
                duration_seconds = None
                if call_start_ts:
                    try:
                        duration_seconds = int(time.time() - call_start_ts)
                    except:
                        duration_seconds = None

                # Inline duration formatting (no helper function)
                if duration_seconds is None:
                    dur_text = "unknown"
                else:
                    m, s = divmod(duration_seconds, 60)
                    dur_text = f"{m}m {s}s" if m else f"{s}s"

                # NEW: If summary was never saved by AI, build fallback summary
                fallback_summary_text = None

                # If AI spoke, we have transcript text
                if not summary_saved and ai_transcript_texts:
                    raw_text = " ".join(ai_transcript_texts)
                    try:
                        completion = client.responses.create(
                            model=OPENAI_MODEL,
                            input=(
                                "Summarise this phone conversation in 5–6 sentences. "
                                "You are an LIC senior advisor. "
                                "Conversation text:\n\n" + raw_text
                            )
                        )
                        fallback_summary_text = completion.output_text.strip()
                        fallback_summary_text += f"\n\nTotal call duration: {dur_text}."
                    except Exception:
                        logger.exception("GPT fallback summary failed")

                
                # If caller was silent
                if not summary_saved and not fallback_summary_text and not had_audio:
                    fallback_summary_text = (
                        f"Caller did not speak anything during the call. "
                        f"Total call duration: {dur_text}."
                )

                # If we still have no summary and there *was* some audio, decide based on duration
                if not summary_saved and not fallback_summary_text and had_audio:
                    # Very short call (e.g. 1–3 seconds) – treat as quick disconnect
                    if  duration_seconds is not None and duration_seconds <= 3:
                            fallback_summary_text = (
                            f"Caller disconnected almost immediately after the call started; "
                            f"no meaningful conversation took place. "
                            f"Total call duration: {dur_text}."
                    )
                    else:
                        # Longer call where AI spoke but didn't return a summary
                        fallback_summary_text = (
                        "Caller did not respond or speak meaningfully during the call. "
                        "The AI agent attempted to speak and prompt the caller, "
                        "but no real conversation took place."
                         f"Total call duration: {dur_text}."
                        )

                # --- END NEW BLOCK ---

                #-----------------------------------

                # Fetch metadata for minimal record
                meta = CALL_TRANSCRIPTS.get(stream_sid) or {}
                meta_call_id = meta.get("call_id") or call_id or (stream_sid or "unknown_call")
                meta_phone = meta.get("phone_number") or caller_number or ""

                # Compute a simple total call duration
                duration_seconds = None
                if call_start_ts:
                    try:
                        duration_seconds = int(time.time() - call_start_ts)
                    except Exception:
                        duration_seconds = None

                def pretty_duration(sec: Optional[int]) -> str:
                    if sec is None:
                        return "unknown"
                    m, s = divmod(sec, 60)
                    if m > 0:
                        return f"{m}m {s}s"
                    return f"{s}s"

                # If we never saw any media frames, the caller literally never spoke
                if not had_audio:
                    summary_text = (
                        f"Call to {meta_phone or 'unknown number'} (call_id={meta_call_id}) "
                        f"ended without the caller speaking anything. "
                        f"Total call duration: {pretty_duration(duration_seconds)}."
                    )
                else:
                    # We had some audio, but no detailed summary from the model
                    summary_text = (
                        f"Call with {meta_phone or 'unknown number'} (call_id={meta_call_id}) "
                        f"ended before a detailed AI summary could be saved. "
                        f"Total call duration: {pretty_duration(duration_seconds)}."
                    )

                # Save minimal record with improved summary (status remains 'stopped')
                # If a fallback summary was generated, override summary_text before inserting
                if not summary_saved and fallback_summary_text:
                    summary_text = fallback_summary_text

                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO call_logs (call_id, phone_number, status, summary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (meta_call_id, meta_phone, "stopped", summary_text),
                )
                conn.commit()
                conn.close()

                save_ppm_outcome(
                    call_id=meta_call_id or "",
                    phone_number=meta_phone or "",
                    outcome_status="stopped",
                    reward_value=0.0,
                    summary=summary_text,
                    duration_seconds=float(duration_seconds or 0),
                )

                save_ppm_debug_event(
                    call_id=meta_call_id or "",
                    stream_sid=stream_sid or "",
                    phone_number=meta_phone or "",
                    event_type="exotel_stop",
                    level="INFO",
                    source="voicebot",
                    message="Call stopped and fallback/final summary saved",
                    payload={
                        "had_audio": had_audio,
                        "summary_saved": summary_saved,
                        "duration_seconds": duration_seconds,
                        "summary_text": summary_text,
                    },
                )


                try:
                    await ppm_log_debug_event_remote({
                        "client_id": "lic_shashinath",
                        "channel": "voice",
                        "call_id": meta_call_id or "",
                        "stream_sid": stream_sid or "",
                        "phone_number": meta_phone or "",
                        "event_type": "exotel_stop",
                        "level": "INFO",
                        "message": "Call stopped and fallback/final summary saved",
                        "payload": {
                            "had_audio": had_audio,
                            "summary_saved": summary_saved,
                            "duration_seconds": duration_seconds,
                            "summary_text": summary_text,
                        },
                    })
                except Exception:
                    logger.exception("Remote PPM log-debug-event failed (exotel_stop)")

                try:
                    await ppm_log_outcome_remote({
                        "client_id": "lic_shashinath",
                        "channel": "voice",
                        "call_id": meta_call_id or "",
                        "phone_number": meta_phone or "",
                        "decision_id": str(meta.get("ppm_decision_id") or ""),
                        "outcome_status": "stopped",
                        "reward_value": 0.0,
                        "summary": summary_text,
                        "duration_seconds": float(duration_seconds or 0),
                    })
                except Exception:
                    logger.exception("Remote PPM log-outcome failed (exotel_stop)")

                # Also log summary to MCP Postgres DB (best-effort)
                try:
                    await log_call_summary_to_db(meta_call_id, meta_phone, summary_text)
                except Exception:
                    logger.exception("Error while calling log_call_summary_to_db from stop event")

                try:
                    ppm_decision_id = meta.get("ppm_decision_id")
                    ppm_strategy_key = meta.get("ppm_strategy_key", "unknown")

                    replied_flag = bool(duration_seconds and duration_seconds > 30)

                    await ppm_log_voice_outcome(
                        decision_id=ppm_decision_id,
                        phone_number=meta_phone,
                        segment="cold",
                        strategy_key=ppm_strategy_key,
                        replied=replied_flag,
                        converted=False,
                        opted_out=False,
                        revenue=0.0,
                        call_duration_seconds=duration_seconds or 0,
                        outcome_notes=summary_text,
                    )
                except Exception:
                    logger.exception("Failed to log outcome to PPM")

                break

    except WebSocketDisconnect:
        logger.info("Exotel WebSocket disconnected")
        save_ppm_debug_event(
            call_id=call_id or "",
            stream_sid=stream_sid or "",
            phone_number=caller_number or "",
            event_type="exotel_websocket_disconnected",
            level="INFO",
            source="voicebot",
            message="Exotel WebSocket disconnected",
        )
    except Exception as e:
        logger.exception("Exception in /exotel-media: %s", e)
        save_ppm_debug_event(
            call_id=call_id or "",
            stream_sid=stream_sid or "",
            phone_number=caller_number or "",
            event_type="exotel_media_exception",
            level="ERROR",
            source="voicebot",
            message=str(e),
        )
    finally:
        if pump_task:
            pump_task.cancel()
        if openai_ws:
            await openai_ws.close()
        if openai_session:
            await openai_session.close()
        # cleanup additive tasks
        if auto_speak_task:
            auto_speak_task.cancel()
        if reengage_task:
            reengage_task.cancel()
        await ws.close()
        

# ---------------------------------------------------------
# Exotel status callback (optional)
# ---------------------------------------------------------

@app.post("/exotel-status")
async def exotel_status(request: Request):
    """
    Optional Exotel status callback to update call_logs.
    """
    form = await request.form()
    logger.info("Exotel status callback: %s", dict(form))
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------
# NEW ADDITIVE: Direct Exotel sequential bulk calling from uploaded Excel
# ---------------------------------------------------------

def exotel_outbound_call_bulk_direct(to_number: str) -> Dict[str, Any]:
    """
    Additive bulk-call helper.
    Uses the Exotel pattern shared by the user, but reads credentials/config from env:
      - EXO_API_KEY
      - EXO_API_TOKEN
      - EXO_CALLER_ID
      - EXOTEL_FLOW_URL or EXO_FLOW_ID
    Only the callee number changes per row.
    """
    exo_api_key = (os.getenv("EXO_API_KEY", "") or "").strip()
    exo_api_token = (os.getenv("EXO_API_TOKEN", "") or "").strip()
    exo_caller_id = (os.getenv("EXO_CALLER_ID", "") or "").strip()
    exo_flow_url = (os.getenv("EXOTEL_FLOW_URL", "") or "").strip()
    exo_flow_id = (os.getenv("EXO_FLOW_ID", "") or "").strip()

    if not exo_flow_url and exo_api_key and exo_flow_id:
        exo_flow_url = f"http://my.exotel.com/{EXO_SID}/exoml/start_voice/{exo_flow_id}"

    if not exo_api_key or not exo_api_token or not exo_caller_id or not exo_flow_url:
        logger.error(
            "Bulk Exotel env missing (EXO_API_KEY / EXO_API_TOKEN / EXO_CALLER_ID / EXOTEL_FLOW_URL or EXO_FLOW_ID)."
        )
        return {
            "error": (
                "Bulk Exotel env missing. Required: EXO_API_KEY, EXO_API_TOKEN, "
                "EXO_CALLER_ID, and EXOTEL_FLOW_URL or EXO_FLOW_ID."
            )
        }

    exotel_url = (
        f"https://{exo_api_key}:{exo_api_token}"
        f"@api.exotel.com/v1/Accounts/{EXO_SID}/Calls/connect.json"
    )
    delay_sec = int(float(os.getenv("BULK_CALL_DELAY_SEC", "240")))
    payload = {
        "From": to_number,
        "CallerId": exo_caller_id,
        "Url": exo_flow_url,
        "TimeLimit": str(delay_sec),
    }

    logger.info("Bulk Exotel outbound call URL: %s", exotel_url)
    logger.info("Bulk Exotel outbound call payload: %s", payload)

    try:
        import requests
        resp = requests.post(
            exotel_url,
            data=payload,
            headers={"accept": "application/json"},
            timeout=20,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            return resp.json()
        return {"raw": resp.text}
    except Exception as e:
        logger.exception("Error placing bulk Exotel outbound call: %s", e)
        return {"error": str(e)}


@app.post("/bulk-call-excel-sequential")
async def bulk_call_excel_sequential(request: Request):
    """
    Additive bulk calling endpoint.
    Rules:
    1. First column of Excel is treated as callee number.
    2. Calls are triggered sequentially.
    3. Delay between two calls is minimum BULK_CALL_DELAY_SEC env variable.
    4. Existing code paths remain untouched.
    """
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse({"error": f"Failed reading form-data: {e}"}, status_code=400)

    file = form.get("file")
    if file is None:
        return JSONResponse({"error": "Please upload an Excel file in form field 'file'."}, status_code=400)

    filename = (getattr(file, "filename", "") or "").lower()
    if not (filename.endswith(".xlsx") or filename.endswith(".xls")):
        return JSONResponse({"error": "Please upload an Excel file (.xlsx or .xls)."}, status_code=400)

    content = await file.read()
    if not content:
        return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)

    import io as _io
    import uuid as _uuid
    import re as _re
    import pandas as _pd
    import asyncio as _asyncio

    try:
        # First column is callee number. Read with no header so row-1 is not lost.
        df = _pd.read_excel(_io.BytesIO(content), header=None)
    except Exception as e:
        logger.exception("Failed reading bulk sequential Excel")
        return JSONResponse({"error": f"Failed to read Excel: {e}"}, status_code=400)

    if df.empty:
        return JSONResponse({"error": "Excel has no rows."}, status_code=400)

    raw_numbers = df.iloc[:, 0].tolist()

    def _clean_bulk_number(x: object) -> str:
        s = "" if x is None else str(x).strip()
        if not s:
            return ""
        # skip obvious headers like 'phone'
        if s.lower() in {"phone", "mobile", "number", "contact", "callee", "callee_number"}:
            return ""
        # remove everything except digits and +
        s = _re.sub(r"[^\d+]", "", s)
        digits = _re.sub(r"\D", "", s)
        if not digits:
            return ""
        if len(digits) == 10:
            return "0" + digits
        if len(digits) == 12 and digits.startswith("91"):
            return "0" + digits[2:]
        if len(digits) == 11 and digits.startswith("0"):
            return digits
        return digits

    numbers = []
    seen = set()
    for item in raw_numbers:
        num = _clean_bulk_number(item)
        if num and num not in seen:
            seen.add(num)
            numbers.append(num)

    if not numbers:
        return JSONResponse({"error": "No valid callee numbers found in the first column."}, status_code=400)

    delay_sec = float(os.getenv("BULK_CALL_DELAY_SEC", "2.0"))
    batch_id = str(_uuid.uuid4())
    results = []
    called_ok = 0
    called_failed = 0

    async with BULK_CALL_LOCK:
        for idx, number in enumerate(numbers, start=1):
            result = await _asyncio.to_thread(exotel_outbound_call_bulk_direct, number)
            if isinstance(result, dict) and result.get("error"):
                called_failed += 1
                results.append({
                    "row_number": idx,
                    "phone": number,
                    "status": "error",
                    "error": result.get("error"),
                })
            else:
                called_ok += 1
                results.append({
                    "row_number": idx,
                    "phone": number,
                    "status": "ok",
                    "result": result,
                })

            if idx < len(numbers) and delay_sec > 0:
                await _asyncio.sleep(delay_sec)

    return {
        "status": "ok",
        "bulk_batch_id": batch_id,
        "phone_column_used": 0,
        "total_numbers": len(numbers),
        "called_ok": called_ok,
        "called_failed": called_failed,
        "delay_sec": delay_sec,
        "results": results,
    }


def _build_bulk_progress_snapshot(batch_id: str) -> Dict[str, Any]:
    progress = BULK_CALL_PROGRESS.get(batch_id) or {}
    return {
        "batch_id": batch_id,
        "status": progress.get("status", "unknown"),
        "total_numbers": progress.get("total_numbers", 0),
        "running_count": progress.get("running_count", 0),
        "completed_count": progress.get("completed_count", 0),
        "failed_count": progress.get("failed_count", 0),
        "current_index": progress.get("current_index", 0),
        "current_number": progress.get("current_number", ""),
        "delay_sec": progress.get("delay_sec", 0),
        "message": progress.get("message", ""),
        "results": progress.get("results", []),
    }


async def _run_bulk_call_excel_sequential_batch(
    *,
    batch_id: str,
    numbers: List[str],
    delay_sec: float,
) -> None:
    progress = BULK_CALL_PROGRESS.get(batch_id)
    if progress is None:
        return

    progress["status"] = "running"
    progress["message"] = "Bulk batch is running."

    async with BULK_CALL_LOCK:
        total = len(numbers)
        for idx, number in enumerate(numbers, start=1):
            progress["current_index"] = idx
            progress["current_number"] = number
            progress["running_count"] = 1
            progress["message"] = f"Calling {number} ({idx}/{total})"

            
            result = await asyncio.to_thread(exotel_outbound_call_bulk_direct, number)
            if isinstance(result, dict) and result.get("error"):
                progress["failed_count"] += 1
                progress["results"].append({
                    "row_number": idx,
                    "phone": number,
                    "status": "error",
                    "error": result.get("error"),
                })
            else:
                progress["completed_count"] += 1
                progress["results"].append({
                    "row_number": idx,
                    "phone": number,
                    "status": "ok",
                    "result": result,
                })

            progress["running_count"] = 0

            if idx < total and delay_sec > 0:
                progress["message"] = f"Waiting {delay_sec} seconds before next call."
                await asyncio.sleep(delay_sec)
                kill_info = await kill_previous_bulk_call_and_wait_until_stopped()
                print("CALL AUTO-KILLED:", kill_info)

    progress["current_number"] = ""
    progress["running_count"] = 0
    progress["status"] = "completed_with_errors" if progress["failed_count"] > 0 else "completed"
    progress["message"] = "Bulk batch finished."


@app.get("/bulk-call-progress/{batch_id}")
async def bulk_call_progress(batch_id: str):
    progress = BULK_CALL_PROGRESS.get(batch_id)
    if not progress:
        return JSONResponse({"error": "batch_id not found"}, status_code=404)
    return _build_bulk_progress_snapshot(batch_id)


@app.post("/bulk-call-excel-sequential-start")
async def bulk_call_excel_sequential_start(request: Request):
    """
    Starts the sequential bulk batch in the background and returns immediately
    so UI can poll progress in real time.
    Existing call logic remains untouched.
    """
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse({"error": f"Failed reading form-data: {e}"}, status_code=400)

    file = form.get("file")
    if file is None:
        return JSONResponse({"error": "Please upload an Excel file in form field 'file'."}, status_code=400)

    filename = (getattr(file, "filename", "") or "").lower()
    if not (filename.endswith(".xlsx") or filename.endswith(".xls")):
        return JSONResponse({"error": "Please upload an Excel file (.xlsx or .xls)."}, status_code=400)

    content = await file.read()
    if not content:
        return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)

    import io as _io
    import uuid as _uuid
    import re as _re
    import pandas as _pd

    try:
        df = _pd.read_excel(_io.BytesIO(content), header=None)
    except Exception as e:
        logger.exception("Failed reading bulk sequential Excel for background batch")
        return JSONResponse({"error": f"Failed to read Excel: {e}"}, status_code=400)

    if df.empty:
        return JSONResponse({"error": "Excel has no rows."}, status_code=400)

    raw_numbers = df.iloc[:, 0].tolist()

    def _clean_bulk_number_start(x: object) -> str:
        s = "" if x is None else str(x).strip()
        if not s:
            return ""
        if s.lower() in {"phone", "mobile", "number", "contact", "callee", "callee_number"}:
            return ""
        s = _re.sub(r"[^\d+]", "", s)
        digits = _re.sub(r"\D", "", s)
        if not digits:
            return ""
        if len(digits) == 10:
            return "0" + digits
        if len(digits) == 12 and digits.startswith("91"):
            return "0" + digits[2:]
        if len(digits) == 11 and digits.startswith("0"):
            return digits
        return digits

    numbers = []
    seen = set()
    for item in raw_numbers:
        num = _clean_bulk_number_start(item)
        if num and num not in seen:
            seen.add(num)
            numbers.append(num)

    if not numbers:
        return JSONResponse({"error": "No valid callee numbers found in the first column."}, status_code=400)

    delay_sec = float(os.getenv("BULK_CALL_DELAY_SEC", "2.0"))
    batch_id = str(_uuid.uuid4())

    BULK_CALL_PROGRESS[batch_id] = {
        "status": "queued",
        "total_numbers": len(numbers),
        "running_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "current_index": 0,
        "current_number": "",
        "delay_sec": delay_sec,
        "message": "Batch queued.",
        "results": [],
    }

    asyncio.create_task(
        _run_bulk_call_excel_sequential_batch(
            batch_id=batch_id,
            numbers=numbers,
            delay_sec=delay_sec,
        )
    )

    return {
        "status": "ok",
        "batch_id": batch_id,
        "total_numbers": len(numbers),
        "delay_sec": delay_sec,
        "message": "Bulk batch started in background.",
    }



# ---------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    logger.info("Starting uvicorn on port %s", port)
    uvicorn.run(
        "ws_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
    )
#----------------------------------------------------
#------add helper  functions for downloading  ranked_customers csv  file  from  call logs db  -----
#-----------------------------------------------------

DB_PATH = "/data/call_logs.db"  # adjust only if you use a different path


def _extract_duration_seconds(summary: str) -> int:
    if not isinstance(summary, str):
        return 0

    m = re.search(r"Total call duration:\s*(\d+)m\s*(\d+)s", summary)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        return minutes * 60 + seconds

    m2 = re.search(r"Total call duration:\s*(\d+)s", summary)
    if m2:
        return int(m2.group(1))

    return 0


def _load_call_logs_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT id, call_id, phone_number, status, summary, created_at FROM call_logs",
            conn,
        )
    finally:
        conn.close()
    return df


def _build_interest_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["summary"] = df["summary"].fillna("")

    # duration
    df["duration_sec"] = df["summary"].apply(_extract_duration_seconds)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(df["summary"])
    text_interest = np.linalg.norm(X.toarray(), axis=1)

    ti_arr = np.array(text_interest, dtype=float)
    max_ti = float(ti_arr.max()) if len(ti_arr) > 0 else 0.0
    ti_norm = ti_arr / max_ti if max_ti > 0 else np.zeros_like(ti_arr)

    dur_arr = df["duration_sec"].values.astype(float)
    max_dur = float(dur_arr.max()) if len(dur_arr) > 0 else 0.0
    dur_norm = dur_arr / max_dur if max_dur > 0 else np.zeros_like(dur_arr)

    # keyword score
    positive_keywords = [
        "policy", "policies", "premium", "premiums", "coverage",
        "retirement", "investment", "invest", "term plan", "pension",
        "ulip", "bonus", "sum assured",
        "policy lene", "policy kharidna", "policy kharid", "interest dikhaya",
        "रुचि", "निवेश",
    ]

    def keyword_score(text: str) -> int:
        t = text.lower()
        return sum(1 for kw in positive_keywords if kw in t)

    kw_scores = np.array([keyword_score(s) for s in df["summary"]], dtype=float)
    max_kw = float(kw_scores.max()) if len(kw_scores) > 0 else 0.0
    kw_norm = kw_scores / max_kw if max_kw > 0 else np.zeros_like(kw_scores)

    interest_score = 0.5 * ti_norm + 0.3 * dur_norm + 0.2 * kw_norm

    df["text_interest"] = ti_norm
    df["duration_norm"] = dur_norm
    df["keyword_norm"] = kw_norm
    df["interest_score"] = interest_score

    return df


def _rank_customers(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    df_sorted = df.sort_values("interest_score", ascending=False)
    grouped = df_sorted.groupby("phone_number")
    best_rows = grouped.head(1).reset_index(drop=True)
    best_rows = best_rows.sort_values("interest_score", ascending=False)
    return best_rows.head(top_k).copy()


# ---------------------------------------------------------
# NEW: Bulk calling from uploaded Excel
# ---------------------------------------------------------

@app.post("/bulk-call-excel")
async def bulk_call_excel(request: Request):
    """
    Bulk call from uploaded Excel (.xlsx/.xls).

    SAFETY / NO-CORRUPTION GUARDBAND:
    - If Exotel creds are missing (EXO_SID / EXO_API_TOKEN / EXO_CALLER_ID), we return 400 and DO NOT write to DB.
    - By default we DO NOT insert any "bulk_triggered" rows into call_logs.
      Set env BULK_CALL_WRITE_AUDIT=1 if you explicitly want audit rows.
    - We treat any {"error": "..."} returned by exotel_outbound_call as a failed call trigger.
      (So you don't get fake called_ok counts.)
    """
    # Parse multipart form and extract `file`
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse({"error": f"Failed reading form-data: {e}"}, status_code=400)

    file = form.get("file")

    # Validate Exotel env (ONLY these names matter)
    if not EXO_SID or not EXO_API_TOKEN or not EXO_CALLER_ID:
        return JSONResponse(
            {
                "error": "Exotel env missing. Required: EXO_SID, EXO_API_TOKEN, EXO_CALLER_ID (and optionally EXO_FLOW_ID / EXOTEL_FLOW_URL)."
            },
            status_code=400,
        )

    # Local imports to avoid touching existing global imports
    import pandas as _pd
    import io as _io
    import asyncio as _asyncio
    import re as _re
    import uuid as _uuid

    filename = (getattr(file, "filename", "") or "").lower()
    if not (filename.endswith(".xlsx") or filename.endswith(".xls")):
        return JSONResponse({"error": "Please upload an Excel file (.xlsx or .xls)."}, status_code=400)

    content = await file.read()
    if not content:
        return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)

    try:
        df = _pd.read_excel(_io.BytesIO(content))
    except Exception as e:
        logger.exception("Failed reading Excel")
        return JSONResponse({"error": f"Failed to read Excel: {e}"}, status_code=400)

    if df.empty:
        return JSONResponse({"error": "Excel has no rows."}, status_code=400)

    # Pick phone/number column
    cols = [str(c) for c in df.columns]
    phone_col = None
    for c in cols:
        cl = c.strip().lower()
        if any(k in cl for k in ["phone", "mobile", "number", "contact"]):
            phone_col = c
            break
    if phone_col is None:
        phone_col = cols[0]

    raw_numbers = df[phone_col].tolist()

    def _clean_number(x: object) -> str:
        s = "" if x is None else str(x)
        s = s.strip()
        if not s:
            return ""
        # keep + and digits; drop everything else
        s = _re.sub(r"[^\d+]", "", s)
        # Excel floats / scientific notation edge case
        if "e" in s.lower():
            try:
                s2 = str(int(float(str(x))))
                s2 = _re.sub(r"[^\d+]", "", s2)
                s = s2
            except Exception:
                pass
        # normalize +91XXXXXXXXXX or 91XXXXXXXXXX or XXXXXXXXXX to 91XXXXXXXXXX
        digits = _re.sub(r"\D", "", s)
        if len(digits) == 10:
            digits = "91" + digits
        elif len(digits) == 12 and digits.startswith("91"):
            pass
        else:
            # leave as-is; exotel may accept other formats; your validation can be stricter later
            return digits
        return digits

    numbers = []
    for rn in raw_numbers:
        n = _clean_number(rn)
        if n:
            numbers.append(n)

    if not numbers:
        return JSONResponse({"error": f"No phone numbers found in column '{phone_col}'."}, status_code=400)

    # De-dup while preserving order
    seen = set()
    deduped = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    numbers = deduped

    delay_sec = float(os.getenv("BULK_CALL_DELAY_SEC", "2.0"))
    write_audit = os.getenv("BULK_CALL_WRITE_AUDIT", "0").strip() == "1"
    batch_id = str(_uuid.uuid4())

    results = []
    called_ok = 0
    called_failed = 0

    async with BULK_CALL_LOCK:
        for n in numbers:
            # Trigger Exotel call (run sync requests.post in a thread to avoid blocking the event loop)
            try:
                r = await _asyncio.to_thread(exotel_outbound_call, n)
                # Treat {"error": "..."} as failure
                if isinstance(r, dict) and r.get("error"):
                    called_failed += 1
                    results.append({"phone": n, "status": "error", "error": r.get("error"), "batch_id": batch_id})
                    if write_audit:
                        try:
                            conn = sqlite3.connect(DB_PATH)
                            cur = conn.cursor()
                            cur.execute(
                                """
                                INSERT INTO call_logs (call_id, phone_number, status, summary)
                                VALUES (?, ?, ?, ?)
                                """,
                                ("", n, "bulk_failed", f"bulk_batch_id={batch_id}; error={r.get('error')}"),
                            )
                            conn.commit()
                            conn.close()
                        except Exception:
                            logger.exception("Failed inserting bulk_failed audit row for %s", n)
                else:
                    called_ok += 1
                    results.append({"phone": n, "status": "ok", "result": r, "batch_id": batch_id})
                    if write_audit:
                        try:
                            conn = sqlite3.connect(DB_PATH)
                            cur = conn.cursor()
                            cur.execute(
                                """
                                INSERT INTO call_logs (call_id, phone_number, status, summary)
                                VALUES (?, ?, ?, ?)
                                """,
                                ("", n, "bulk_triggered", f"bulk_batch_id={batch_id}; bulk call triggered."),
                            )
                            conn.commit()
                            conn.close()
                        except Exception:
                            logger.exception("Failed inserting bulk_triggered audit row for %s", n)
            except Exception as e:
                logger.exception("Bulk call failed for %s", n)
                called_failed += 1
                results.append({"phone": n, "status": "error", "error": str(e), "batch_id": batch_id})
                if write_audit:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        cur.execute(
                            """
                            INSERT INTO call_logs (call_id, phone_number, status, summary)
                            VALUES (?, ?, ?, ?)
                            """,
                            ("", n, "bulk_failed", f"bulk_batch_id={batch_id}; exception={str(e)}"),
                        )
                        conn.commit()
                        conn.close()
                    except Exception:
                        logger.exception("Failed inserting bulk_failed audit row for %s", n)

            if delay_sec > 0:
                await _asyncio.sleep(delay_sec)
    return {
        "status": "ok",
        "bulk_batch_id": batch_id,
        "phone_column_used": phone_col,
        "total_numbers": len(numbers),
        "called_ok": called_ok,
        "called_failed": called_failed,
        "delay_sec": delay_sec,
        "write_audit": write_audit,
        "results": results,
    }


# ===================== ADDITIVE BULK CALL CONTROL =====================

import time

ACTIVE_BULK_CALL = {
    "call_id": "",
    "phone_number": "",
    "stream_sid": "",
    "ws": None,
    "openai_ws": None,
    "started_at": 0.0,
}

ACTIVE_BULK_CALL_LOCK = asyncio.Lock()

async def kill_previous_bulk_call_and_wait_until_stopped(
    max_wait_sec: float = 30.0,
    poll_interval_sec: float = 0.5,
) -> Dict[str, Any]:
    start_wait = time.time()

    while True:
        async with ACTIVE_BULK_CALL_LOCK:
            active_ws = ACTIVE_BULK_CALL.get("ws")
            active_openai_ws = ACTIVE_BULK_CALL.get("openai_ws")
            active_call_id = ACTIVE_BULK_CALL.get("call_id") or ""
            active_phone = ACTIVE_BULK_CALL.get("phone_number") or ""
            active_stream_sid = ACTIVE_BULK_CALL.get("stream_sid") or ""

        has_live_call = False

        try:
            if active_ws is not None:
                has_live_call = True
        except Exception:
            pass

        try:
            if active_openai_ws is not None and not getattr(active_openai_ws, "closed", True):
                has_live_call = True
        except Exception:
            pass

        if not has_live_call:
            return {"status": "no_active_call"}

        logger.warning(
            "Previous bulk call still active. Killing and waiting. call_id=%s phone=%s stream_sid=%s",
            active_call_id,
            active_phone,
            active_stream_sid,
        )

        try:
            if active_openai_ws is not None and not getattr(active_openai_ws, "closed", True):
                await active_openai_ws.close()
        except Exception:
            logger.exception("Failed closing previous active bulk OpenAI WS")

        try:
            if active_ws is not None:
                await active_ws.close()
        except Exception:
            logger.exception("Failed closing previous active bulk Exotel WS")

        waited = time.time() - start_wait
        if waited >= max_wait_sec:
            async with ACTIVE_BULK_CALL_LOCK:
                ACTIVE_BULK_CALL["call_id"] = ""
                ACTIVE_BULK_CALL["phone_number"] = ""
                ACTIVE_BULK_CALL["stream_sid"] = ""
                ACTIVE_BULK_CALL["ws"] = None
                ACTIVE_BULK_CALL["openai_ws"] = None
                ACTIVE_BULK_CALL["started_at"] = 0.0

            return {
                "status": "force_cleared_after_timeout",
                "call_id": active_call_id,
                "phone_number": active_phone,
                "stream_sid": active_stream_sid,
                "waited_sec": waited,
            }

        await asyncio.sleep(poll_interval_sec)
# ===================== END ADDITIVE =====================
