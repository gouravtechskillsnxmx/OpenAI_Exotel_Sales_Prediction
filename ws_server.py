"""
ws_server.py — Exotel Outbound Realtime LIC Agent + Call Logs + Leads + MCP

Features:
- Outbound calls via Exotel Connect API to a Voicebot App/Flow (EXO_FLOW_ID)
- Realtime LIC insurance agent voicebot using OpenAI Realtime
- MCP integration (LIC_CRM_MCP_BASE_URL) for save_call_summary tool
- Exotel status webhook saving call details into SQLite
- Leads table + CSV upload to trigger outbound calls
- Simple dashboard at /dashboard:
  - Upload CSV (name,phone) to trigger outbound calls
  - View recent call logs

ENV (set in Render):
  EXO_SID           e.g. gouravnxmx1
  EXO_API_KEY       from Exotel API settings
  EXO_API_TOKEN     from Exotel API settings
  EXO_FLOW_ID       e.g. 1077390 (your Voicebot app id)
  EXO_SUBDOMAIN     api or api.in   (NOT the full domain)
  EXO_CALLER_ID     your Exophone, e.g. 09513886363

  OPENAI_API_KEY or OpenAI_Key or OPENAI_KEY
  OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview (recommended)

  PUBLIC_BASE_URL   e.g. openai-exotel-elevenlabs-outbound.onrender.com
  LOG_LEVEL=INFO

  DB_PATH=/tmp/call_logs.db   (or /data/call_logs.db if you have persistent disk)
  SAVE_TTS_WAV=1              (if you want to write TTS wav files under /tmp)

  LIC_CRM_MCP_BASE_URL=https://lic-crm-mcp.onrender.com    (MCP server for saving call summaries)
"""

import asyncio
import base64
import csv
import io
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Optional, List

import audioop
import httpx
import requests
from aiohttp import ClientSession, WSMsgType
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
    UploadFile,
    File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# Optional: these were in your original file; keep if installed
import numpy as np  # noqa: F401
from scipy.signal import resample  # noqa: F401

# ---------------- Logging ----------------
level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, level, logging.INFO))
logger = logging.getLogger("ws_server")

# ---------------- Global transcript store ----------------
CALL_TRANSCRIPTS = {}

# ---------------- DB (SQLite) ----------------
DB_PATH = os.getenv("DB_PATH", "/tmp/call_logs.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # call logs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid TEXT,
            from_number TEXT,
            to_number TEXT,
            status TEXT,
            recording_url TEXT,
            started_at TEXT,
            ended_at TEXT,
            raw_payload TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # leads
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT,
            status TEXT,
            last_call_sid TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    conn.commit()
    conn.close()
    logger.info("SQLite DB initialized at %s", DB_PATH)


init_db()

# ---------------- FastAPI app ----------------
app = FastAPI(title="Outbound LIC Voicebot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Exotel Helpers ----------------


def exotel_call_url() -> str:
    """
    Compose Exotel Connect API URL:
      https://{subdomain}.exotel.com/v1/Accounts/{sid}/Calls/connect
    e.g. https://api.exotel.com/v1/Accounts/gouravnxmx1/Calls/connect
    """
    sub = os.getenv("EXO_SUBDOMAIN", "api")  # "api" or "api.in"
    sid = os.getenv("EXO_SID", "")
    return f"https://{sub}.exotel.com/v1/Accounts/{sid}/Calls/connect"


def exotel_headers_auth():
    """
    Return basic auth (username, password) for Exotel.
    """
    api_key = os.getenv("EXO_API_KEY", "")
    api_token = os.getenv("EXO_API_TOKEN", "")
    return api_key, api_token


# ---------------- Audio helpers ----------------


def downsample_24k_to_8k_pcm16(pcm24: bytes) -> bytes:
    """24 kHz mono PCM16 -> 8 kHz mono PCM16 using stdlib audioop."""
    converted, _ = audioop.ratecv(pcm24, 2, 1, 24000, 8000, None)
    return converted


def upsample_8k_to_24k_pcm16(pcm8: bytes) -> bytes:
    """8 kHz mono PCM16 -> 24 kHz mono PCM16 using stdlib audioop."""
    converted, _ = audioop.ratecv(pcm8, 2, 1, 8000, 24000, None)
    return converted


# ---------------- Bootstrap for Exotel Voicebot ----------------

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip()  # no protocol


@app.get("/exotel-ws-bootstrap")
async def exotel_ws_bootstrap():
    """
    Called by Exotel Voicebot applet (Dynamic WebSocket URL).
    Returns the wss:// URL pointing back to this service's /exotel-media route.
    """
    try:
        logger.info("0.1 PUBLIC_BASE_URL=%s", PUBLIC_BASE_URL)
        # IMPORTANT: default host aligned with working version
        base = PUBLIC_BASE_URL or "openai-exotel-elevenlabs-outbound.onrender.com"
        url = f"wss://{base}/exotel-media"
        logger.info("0.2")
        logger.info("Bootstrap served: %s", url)
        return {"url": url}
    except Exception as e:
        logger.exception("/exotel-ws-bootstrap error: %s", e)
        fallback = PUBLIC_BASE_URL or "openai-exotel-elevenlabs-outbound.onrender.com"
        return {"url": f"wss://{fallback}/exotel-media"}


# ---------------- Helper: Exotel outbound (Connect API) ----------------


def exotel_outbound_call(to_number: str, caller_id: Optional[str] = None) -> dict:
    """
    Trigger an outbound call via Exotel Connect API.
    'to_number' is the customer's phone, 'caller_id' is your Exophone.

    Returns parsed JSON from Exotel, or {"raw": text} on non-JSON response.
    """
    logger.info("Step 1.2")
    url = exotel_call_url()
    auth = exotel_headers_auth()
    flow_id = os.getenv("EXO_FLOW_ID", "")
    if not flow_id:
        raise RuntimeError("EXO_FLOW_ID is not set")

    if not caller_id:
        caller_id = os.getenv("EXO_CALLER_ID", "")

    payload = {
        "From": to_number,
        "To": caller_id,  # your Exophone
        "CallerId": caller_id,
        "Url": f"http://my.exotel.com/Exotel/exoml/start/{flow_id}",
    }
    logger.info("Step 1.3")
    logger.info("Exotel outbound call payload: %s", payload)
    resp = requests.post(url, data=payload, auth=auth, timeout=30)
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    logger.info("Exotel outbound call result: %s", data)
    logger.info("Step 1.4")
    return data


# ---------------- OpenAI / MCP ENV ----------------
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OpenAI_Key")
    or os.getenv("OPENAI_KEY", "")
)
# Default changed to gpt-4o-realtime-preview as discussed
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

LIC_CRM_MCP_BASE_URL = os.getenv("LIC_CRM_MCP_BASE_URL", "").rstrip("/")

SAVE_TTS_WAV = bool(int(os.getenv("SAVE_TTS_WAV", "0")))


def public_url(path: str) -> str:
    host = PUBLIC_BASE_URL
    if not host:
        return ""
    return f"https://{host.rstrip('/')}{path}"


# ---------------- TTS placeholder (optional) ----------------

def make_tts(text: str) -> bytes:
    """
    Placeholder TTS: for now we are streaming audio from Realtime.
    This is kept only if you later want pre-generated TTS.
    """
    logger.info("make_tts called with text: %s", text)
    return b""


# ---------------- Outbound call API (HTTP) ----------------


class OutboundCallRequest(BaseModel):
    to_number: str
    caller_name: Optional[str] = None


@app.post("/exotel-outbound-call")
async def exotel_outbound_call_api(req: OutboundCallRequest):
    """
    Trigger an outbound call via Exotel Connect API.
    Exotel Voicebot flow should eventually connect WebSocket to /exotel-media.
    """
    try:
        logger.info("Step 1")
        result = exotel_outbound_call(req.to_number)
        logger.info("Step 1.1")
        return JSONResponse({"status": "ok", "exotel": result})
    except Exception as e:
        logger.exception("Error placing outbound call to Exotel")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


# ---------------- Exotel status webhook ----------------
@app.post("/exotel-status")
async def exotel_status(request: Request):
    """
    Exotel status webhook:
      - Called at different stages (ringing, answered, completed).
      - Saves basic call details into SQLite call_logs.
    """
    form = await request.form()
    data = dict(form)
    logger.info("Exotel status webhook payload: %s", data)

    call_sid = data.get("CallSid") or data.get("Sid") or ""
    frm = data.get("From") or data.get("From[]") or ""
    to = data.get("To") or data.get("To[]") or ""
    status = data.get("Status") or data.get("Status[]") or ""
    recording_url = (
        data.get("RecordingUrl")
        or data.get("RecordingUrl[]")
        or data.get("RecordingURL")
        or ""
    )
    started_at = data.get("StartTime") or data.get("StartTime[]") or ""
    ended_at = data.get("EndTime") or data.get("EndTime[]") or ""

    raw_payload = json.dumps(data, ensure_ascii=False)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO call_logs (
            call_sid, from_number, to_number, status,
            recording_url, started_at, ended_at, raw_payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (call_sid, frm, to, status, recording_url, started_at, ended_at, raw_payload),
    )
    conn.commit()
    conn.close()

    return PlainTextResponse("OK")


# ---------------- Simple dashboard (logs + CSV upload) ----------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>LIC Outbound Calls Dashboard</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
    }
    h1, h2 {
      color: #333;
    }
    .section {
      border: 1px solid #ccc;
      padding: 16px;
      margin-bottom: 24px;
      border-radius: 8px;
    }
    table {
      border-collapse: collapse;
      width: 100%%;
      margin-top: 12px;
    }
    table, th, td {
      border: 1px solid #ccc;
    }
    th, td {
      padding: 8px;
      text-align: left;
    }
    .btn {
      display: inline-block;
      padding: 8px 12px;
      background: #007bff;
      color: #fff;
      border-radius: 4px;
      text-decoration: none;
      border: none;
      cursor: pointer;
    }
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .status-badge {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 12px;
      font-size: 0.85rem;
      color: #fff;
    }
    .status-initiated { background: #6c757d; }
    .status-ringing   { background: #17a2b8; }
    .status-answered  { background: #28a745; }
    .status-completed { background: #007bff; }
    .status-failed    { background: #dc3545; }
  </style>
</head>
<body>
  <h1>LIC Outbound Calls Dashboard</h1>

  <div class="section">
    <h2>Upload Leads CSV (name,phone)</h2>
    <form action="/upload-leads" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required />
      <button class="btn" type="submit">Upload & Schedule Calls</button>
    </form>
    <p>CSV format: <code>name,phone</code> (header row optional)</p>
  </div>

  <div class="section">
    <h2>Recent Call Logs</h2>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Call SID</th>
          <th>From</th>
          <th>To</th>
          <th>Status</th>
          <th>Recording</th>
          <th>Started</th>
          <th>Ended</th>
          <th>Created</th>
        </tr>
      </thead>
      <tbody>
        %s
      </tbody>
    </table>
  </div>
</body>
</html>
"""


@app.get("/dashboard")
async def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, call_sid, from_number, to_number, status, recording_url, started_at, ended_at, created_at FROM call_logs ORDER BY id DESC LIMIT 50"
    )
    rows = cur.fetchall()
    conn.close()

    body_rows = []
    for row in rows:
        (
            cid,
            call_sid,
            from_number,
            to_number,
            status,
            recording_url,
            started_at,
            ended_at,
            created_at,
        ) = row

        status_lower = (status or "").lower()
        status_class = "status-initiated"
        if "ring" in status_lower:
            status_class = "status-ringing"
        elif "answer" in status_lower:
            status_class = "status-answered"
        elif "complete" in status_lower:
            status_class = "status-completed"
        elif "fail" in status_lower:
            status_class = "status-failed"

        if recording_url:
            rec_link = f'<a href="{recording_url}" target="_blank">Play</a>'
        else:
            rec_link = ""

        body_rows.append(
            f"""
            <tr>
              <td>{cid}</td>
              <td>{call_sid}</td>
              <td>{from_number}</td>
              <td>{to_number}</td>
              <td><span class="status-badge {status_class}">{status}</span></td>
              <td>{rec_link}</td>
              <td>{started_at}</td>
              <td>{ended_at}</td>
              <td>{created_at}</td>
            </tr>
            """
        )

    html = DASHBOARD_HTML % "\n".join(body_rows)
    return HTMLResponse(html)


@app.post("/upload-leads")
async def upload_leads(file: UploadFile = File(...)):
    """
    Upload CSV of leads (name,phone) and trigger outbound calls.
    Very simple demo: we place calls sequentially (no scheduling).
    """
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    f = io.StringIO(text)
    reader = csv.reader(f)

    leads = []
    for row in reader:
        if not row or len(row) < 2:
            continue
        name, phone = row[0].strip(), row[1].strip()
        if not name or not phone:
            continue
        if name.lower() == "name" and phone.lower() == "phone":
            continue
        leads.append((name, phone))

    logger.info("Parsed %d leads from CSV", len(leads))

    results = []
    for name, phone in leads:
        try:
            # Insert lead into DB
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO leads (name, phone, status)
                VALUES (?, ?, ?)
                """,
                (name, phone, "pending"),
            )
            lead_id = cur.lastrowid
            conn.commit()
            conn.close()

            # Trigger Exotel call
            result = exotel_outbound_call(phone)
            call_sid = result.get("Call", {}).get("Sid") if isinstance(result, dict) else None

            # Update lead record with call_sid, status
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE leads
                SET last_call_sid = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (call_sid or "", "initiated", lead_id),
            )
            conn.commit()
            conn.close()

            results.append({"name": name, "phone": phone, "status": "ok", "call_sid": call_sid})
        except Exception as e:
            logger.exception("Error handling lead %s (%s)", name, phone)
            results.append({"name": name, "phone": phone, "status": "error", "error": str(e)})

    return JSONResponse({"status": "ok", "results": results})


# ---------------- Realtime media bridge (Exotel <-> OpenAI via MCP) ----------------
@app.websocket("/exotel-media")
async def exotel_media_ws(ws: WebSocket):
    """
    Exotel <-> OpenAI Realtime bridge for outbound LIC agent (Shashinath Thakur),
    using MCP tool-calls (save_call_summary) instead of local summariser.
    """
    await ws.accept()
    logger.info("Exotel WS connected (Shashinath LIC agent, realtime)")

    # --- Call metadata (per stream) ---
    call_id: Optional[str] = None        # Exotel CallSid
    caller_number: Optional[str] = None  # customer phone
    stream_sid: Optional[str] = None     # Exotel stream ID
    # -----------------------------------

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY; closing Exotel stream.")
        await ws.close()
        return

    logger.info("Using realtime model: %s", REALTIME_MODEL)

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
        Send 8kHz PCM16 audio back to Exotel as media frames.
        Uses the current stream_sid and sequence counters.
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
        Connect to OpenAI Realtime, configure LIC persona + MCP server,
        and start the pump() loop that sends audio back to Exotel.
        """
        nonlocal openai_session, openai_ws, pump_task

        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            }

            url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

            openai_session = ClientSession()
            logger.info("Connecting to OpenAI Realtime WS at %s ...", url)
            openai_ws = await openai_session.ws_connect(url, headers=headers)
            logger.info("Connected to OpenAI WS")

            # ---------------- LIC persona + MCP instructions ----------------
            instructions_text = (
                "You are Mr. Shashinath Thakur, a senior LIC insurance agent from India. "
                "You speak in friendly Hinglish (mix of Hindi and English), calm and trustworthy, "
                "like a real experienced LIC advisor.\n\n"
                f"This call metadata:\n- call_id = {conn_call_id}\n- phone_number = {conn_caller_number}\n\n"
                "GOALS DURING CALL:\n"
                "1. Greet the caller warmly and clearly introduce yourself as 'LIC agent Mr. Shashinath Thakur'.\n"
                "2. Ask short, clear questions to understand their LIC needs (term plan, money-back, child plan, etc.).\n"
                "3. Explain policies simply: premium, cover amount, term, tax benefit, riders, and claim process.\n"
                "4. Always keep answers short (1–2 sentences) and then ask ONE follow-up question, "
                "then wait silently for the caller to speak.\n"
                "5. Never talk about topics outside LIC insurance and basic financial planning.\n\n"
                "MCP TOOL USAGE (VERY IMPORTANT):\n"
                "- You have a tool `save_call_summary` available via an MCP server.\n"
                "- When the phone call is clearly ending (final goodbye), you MUST:\n"
                "  (a) Infer: intent, interest_score (0–10), next_action, and a 3–6 sentence raw_summary.\n"
                "  (b) Call `save_call_summary` exactly once with:\n"
                "      call_id, phone_number, customer_name (if known), interest_score, intent, next_action, raw_summary.\n"
                "Do NOT call save_call_summary in the middle of the conversation; only once at the end."
            )
            # ----------------------------------------------------------------

            session_config: dict = {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "alloy",
                "turn_detection": {"type": "server_vad"},
                "instructions": instructions_text,
            }

            # Attach MCP server if configured
            if LIC_CRM_MCP_BASE_URL:
                session_config["mcp_servers"] = [
                    {
                        "id": "lic-crm-mcp",
                        "url": f"{LIC_CRM_MCP_BASE_URL}/mcp",
                    }
                ]
            else:
                logger.warning("LIC_CRM_MCP_BASE_URL not set; MCP tools unavailable for this call")

            # Send initial session.update
            await send_openai(
                {
                    "type": "session.update",
                    "session": session_config,
                }
            )
            logger.info("Sent session.update with LIC persona + MCP config")

            # Ask the model to start the first greeting turn
            await send_openai(
                {
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "Start the call now: greet the caller, introduce yourself as LIC agent "
                            "Mr. Shashinath Thakur, and ask how you can help with LIC today."
                        )
                    },
                }
            )

            async def pump():
                """
                Receive events from OpenAI Realtime and forward audio deltas to Exotel.
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

                        # Audio deltas from model
                        if et in ("response.audio.delta", "response.output_audio.delta"):
                            delta = evt.get("delta") or evt.get("audio") or {}
                            b64 = delta.get("audio") or delta.get("data")
                            if not b64:
                                continue

                            try:
                                pcm24 = base64.b64decode(b64)
                            except Exception:
                                logger.exception("Failed to decode audio delta")
                                continue

                            # Downsample 24kHz -> 8kHz for Exotel
                            try:
                                pcm8 = downsample_24k_to_8k_pcm16(pcm24)
                            except Exception:
                                logger.exception("Downsampling 24k -> 8k failed")
                                continue

                            await send_audio_to_exotel(pcm8)

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
                start_obj = evt.get("start") or {}
                stream_sid = start_obj.get("stream_sid") or evt.get("stream_sid")
                start_ts = time.time()

                call_id = start_obj.get("call_sid") or start_obj.get("callSid") or evt.get("call_sid")
                caller_number = (
                    start_obj.get("from")
                    or start_obj.get("caller_id")
                    or start_obj.get("caller_number")
                    or evt.get("from")
                )

                logger.info(
                    "Exotel stream started, stream_sid=%s, call_id=%s, from=%s",
                    stream_sid,
                    call_id,
                    caller_number,
                )

                # Optional: init transcript store for this call (can fill later)
                CALL_TRANSCRIPTS[stream_sid] = {
                    "call_id": call_id,
                    "phone_number": caller_number,
                    "turns": [],  # list of (speaker, text)
                }

                if not openai_started:
                    openai_started = True
                    await connect_openai(call_id or "unknown_call", caller_number or "")

            elif ev == "media":
                # Caller audio (8kHz PCM16) -> upsample to 24kHz -> send to OpenAI
                media = evt.get("media") or {}
                payload_b64 = media.get("payload")
                if payload_b64 and openai_ws and not openai_ws.closed:
                    try:
                        pcm8 = base64.b64decode(payload_b64)
                    except Exception:
                        logger.warning("Invalid base64 in Exotel media payload")
                        continue

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
                logger.info("Exotel sent stop; instructing model to summarise via MCP tool-call and closing WS.")

                # Fetch metadata
                meta = CALL_TRANSCRIPTS.get(stream_sid) or {}
                meta_call_id = meta.get("call_id") or call_id or (stream_sid or "unknown_call")
                meta_phone = meta.get("phone_number") or caller_number or ""

                # Ask model to summarise and call save_call_summary once
                await send_openai(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "The phone call has now ended. "
                                        "Please summarise the entire call and then invoke the "
                                        "`save_call_summary` tool exactly once with:\n"
                                        f"- call_id = {meta_call_id}\n"
                                        f"- phone_number = {meta_phone}\n"
                                        "- customer_name (if known)\n"
                                        "- interest_score (0–10)\n"
                                        "- intent label (buy_term, renew, info_only, not_interested, other)\n"
                                        "- next_action (follow_up, whatsapp_quote, no_contact, other)\n"
                                        "- raw_summary (3–6 sentences in natural language)\n"
                                        "Do not speak this summary aloud; only call the tool."
                                    ),
                                }
                            ],
                        },
                    }
                )
                await send_openai({"type": "response.create"})

                # Clean up local memory
                if stream_sid in CALL_TRANSCRIPTS:
                    CALL_TRANSCRIPTS.pop(stream_sid, None)

                break

            else:
                logger.warning("Unhandled Exotel event: %s", ev)

    except WebSocketDisconnect:
        logger.info("Exotel WS disconnected")
    except Exception as e:
        logger.exception("Error in exotel_media_ws: %s", e)
    finally:
        if pump_task:
            pump_task.cancel()
        try:
            if openai_ws and not openai_ws.closed:
                await openai_ws.close()
        except Exception:
            pass
        try:
            if openai_session:
                await openai_session.close()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        # Summary is handled by MCP save_call_summary tool.

# --------------- Simple index ---------------
@app.get("/")
async def index():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html>
<head><title>Outbound LIC Voicebot</title></head>
<body>
  <h1>Outbound LIC Voicebot</h1>
  <p>This service exposes:</p>
  <ul>
    <li><code>POST /exotel-outbound-call</code> – trigger outbound call</li>
    <li><code>POST /exotel-status</code> – Exotel status webhook</li>
    <li><code>/exotel-media</code> – Exotel WebSocket media &lt;-&gt; OpenAI Realtime</li>
    <li><code>/dashboard</code> – basic call logs dashboard</li>
  </ul>
</body>
</html>
    """
    )


# --------------- Run locally ---------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ws_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        reload=False,
        workers=1,
    )
