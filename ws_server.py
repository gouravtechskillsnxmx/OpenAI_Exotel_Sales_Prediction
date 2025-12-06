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
import os
import sqlite3
import time
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

EXOTEL_SID = os.getenv("EXOTEL_SID", "")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN", "")
EXO_SUBDOMAIN = os.getenv("EXO_SUBDOMAIN", "api")  # kept for compatibility
EXO_CALLER_ID = os.getenv("EXO_CALLER_ID", "")

# New: outbound flow URL (from Exotel support pattern)
EXOTEL_FLOW_URL = os.getenv(
    "EXOTEL_FLOW_URL",
    "http://my.exotel.com/gouravnxmx1/exoml/start_voice/1077390",  # default based on support snippet
)

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OpenAI_Key")
    or os.getenv("OPENAI_KEY", "")
)
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

LIC_CRM_MCP_BASE_URL = os.getenv("LIC_CRM_MCP_BASE_URL", "").rstrip("/")

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip()  # no protocol

DB_PATH = os.getenv("DB_PATH", "/tmp/call_logs.db")


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

# In-memory call transcripts keyed by Exotel stream_sid
CALL_TRANSCRIPTS: Dict[str, Dict[str, Any]] = {}

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
      <h2>Call Logs (Last 50)</h2>
      <button id="refresh-logs">Refresh Logs</button>
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
              <td>${row.created_at || ""}</td>
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

      document.getElementById("single-call-form")
        .addEventListener("submit", triggerSingleCall);

      document.getElementById("refresh-logs")
        .addEventListener("click", loadCallLogs);

      document.getElementById("mcp-test-button")
        .addEventListener("click", runMcpTest);

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
    if not EXOTEL_SID or not EXOTEL_TOKEN or not EXO_CALLER_ID:
        logger.error("Exotel credentials or caller ID missing; cannot place outbound call.")
        return {"error": "exotel credentials/caller id missing"}

    exotel_url = f"https://{EXOTEL_SID}:{EXOTEL_TOKEN}@api.exotel.com/v1/Accounts/{EXOTEL_SID}/Calls/connect.json"

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

            # Build instructions for LIC agent persona
            instructions_text = (
                "You are Mr. Shashinath Thakur, a highly experienced LIC insurance agent "
                "calling from LIC's Mumbai branch. Your job is to:\n"
                "1. Greet the customer warmly in Hindi or Hinglish.\n"
                "2. take his/her permission to speak about LIC policies for 5 mins. ,Confirm you are calling about LIC policies.\n"
                "3. Ask a few probing questions about their existing insurance, "
                "   family, financial goals, and risk appetite.\n"
                "4. Recommend suitable LIC plans (e.g., term, endowment, ULIP, pension) "
                "   with simple explanation (no jargon).\n"
                "5. Be concise, polite, and not pushy.\n"
                "6. At the end, summarise the conversation: what you understood, "
                "   what you recommended, and any next steps.\n\n"
                "VERY IMPORTANT:\n"
                "- After the conversation is finished, you MUST call the tool "
                "  'save_call_summary' exactly once.\n"
                "- In that tool call, fill:\n"
                "    call_id: the call id I have for this phone call,\n"
                "    phone_number: the caller's phone number,\n"
                "    summary: 4–6 sentences summarising the customer's needs, "
                "             what you discussed, which LIC plans you suggested, "
                "             and the next action.\n"
                "- Do not skip the 'save_call_summary' tool call. If you already called it, "
                "  do not call it again.\n"
                "- Always speak naturally, as if on a real phone call.\n"
                "- Use short sentences; pause to let the customer speak.\n"
                "- If the customer asks off-topic questions, gently bring them back to LIC.\n"
            )

            tools_spec = [
                {
                    "type": "function",
                    "name": "save_call_summary",
                    "description": "Persist a structured LIC call summary into the CRM.",
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

            # Ask the model to start the first greeting turn
            await send_openai(
                {
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "Start the call now: greet the caller, introduce yourself as LIC agent "
                            "Mr. Shashinath Thakur, and ask how you can help with LIC today."
                        ),
                        # Force audio output, not just text
                        "modalities": ["text", "audio"],
                    },
                }
            )

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

                except Exception as e:
                    logger.exception("Pump error: %s", e)

            pump_task = asyncio.create_task(pump())

        except Exception as e:
            logger.exception("OpenAI connection error: %s", e)

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
                    except Exception:
                        logger.warning("Invalid base64 in Exotel media payload")
                        continue
                    
                    had_audio = True

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

                # Also log summary to MCP Postgres DB (best-effort)
                try:
                    await log_call_summary_to_db(meta_call_id, meta_phone, summary_text)
                except Exception:
                    logger.exception("Error while calling log_call_summary_to_db from stop event")

                break

    except WebSocketDisconnect:
        logger.info("Exotel WebSocket disconnected")
    except Exception as e:
        logger.exception("Exception in /exotel-media: %s", e)
    finally:
        if pump_task:
            pump_task.cancel()
        if openai_ws:
            await openai_ws.close()
        if openai_session:
            await openai_session.close()
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
