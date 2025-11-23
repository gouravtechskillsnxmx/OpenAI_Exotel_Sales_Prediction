"""
ws_server.py — Exotel Outbound Realtime LIC Agent + Call Logs + Leads + MCP

Features:
- Outbound calls via Exotel Connect API to a Voicebot App/Flow (EXO_FLOW_ID)
- Realtime LIC insurance agent voicebot using OpenAI Realtime
- MCP-backed call summary using Realtime function tool-calls (save_call_summary)
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
  OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview   (recommended)

  PUBLIC_BASE_URL   e.g. openai-exotel-sales-prediction.onrender.com
  LOG_LEVEL=DEBUG   (for full trace) or INFO

  DB_PATH=/tmp/call_logs.db   (or /data/call_logs.db if you have persistent disk)

  LIC_CRM_MCP_BASE_URL=https://lic-crm-mcp.onrender.com    (MCP server; we call /test-save)
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
from typing import Optional, Dict, Any, List

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

# ---------------- Logging ----------------
level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, level, logging.INFO))
logger = logging.getLogger("ws_server")

# ---------------- Global transcript store ----------------
CALL_TRANSCRIPTS: Dict[str, Dict[str, Any]] = {}

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
        logger.info("LOG_POINT_001: /exotel-ws-bootstrap hit, PUBLIC_BASE_URL=%s", PUBLIC_BASE_URL)
        base = PUBLIC_BASE_URL or "openai-exotel-elevenlabs-outbound.onrender.com"
        url = f"wss://{base}/exotel-media"
        logger.info("LOG_POINT_002: Bootstrap served URL=%s", url)
        return {"url": url}
    except Exception as e:
        logger.exception("LOG_POINT_003: /exotel-ws-bootstrap error: %s", e)
        fallback = PUBLIC_BASE_URL or "openai-exotel-elevenlabs-outbound.onrender.com"
        return {"url": f"wss://{fallback}/exotel-media"}


# ---------------- Helper: Exotel outbound (Connect API) ----------------


def exotel_outbound_call(to_number: str, caller_id: Optional[str] = None) -> dict:
    """
    Trigger an outbound call via Exotel Connect API.
    'to_number' is the customer's phone, 'caller_id' is your Exophone.

    Returns parsed JSON from Exotel, or {"raw": text} on non-JSON response.
    """
    logger.info("LOG_POINT_100: exotel_outbound_call called to_number=%s", to_number)
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
    logger.info("LOG_POINT_101: Exotel outbound call payload: %s", payload)
    resp = requests.post(url, data=payload, auth=auth, timeout=30)
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    logger.info("LOG_POINT_102: Exotel outbound call result: %s", data)
    return data


# ---------------- OpenAI / MCP ENV ----------------
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OpenAI_Key")
    or os.getenv("OPENAI_KEY", "")
)
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

LIC_CRM_MCP_BASE_URL = os.getenv("LIC_CRM_MCP_BASE_URL", "").rstrip("/")


# ---------------- TTS placeholder (optional) ----------------

def make_tts(text: str) -> bytes:
    logger.info("make_tts called with text: %s", text)
    return b""


# ---------------- Outbound call API (HTTP) ----------------


class OutboundCallRequest(BaseModel):
    to_number: str
    caller_name: Optional[str] = None


@app.post("/exotel-outbound-call")
async def exotel_outbound_call_api(req: OutboundCallRequest):
    try:
        logger.info("LOG_POINT_110: /exotel-outbound-call hit, body=%s", req.dict())
        result = exotel_outbound_call(req.to_number)
        return JSONResponse({"status": "ok", "exotel": result})
    except Exception as e:
        logger.exception("LOG_POINT_111: Error placing outbound call to Exotel")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


# ---------------- Exotel status webhook ----------------
@app.post("/exotel-status")
async def exotel_status(request: Request):
    form = await request.form()
    data = dict(form)
    logger.info("LOG_POINT_120: Exotel status webhook payload: %s", data)

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
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1, h2 { color: #333; }
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
    table, th, td { border: 1px solid #ccc; }
    th, td { padding: 8px; text-align: left; }
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
        "SELECT id, call_sid, from_number, to_number, status, recording_url, "
        "started_at, ended_at, created_at FROM call_logs ORDER BY id DESC LIMIT 50"
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

        rec_link = f'<a href="{recording_url}" target="_blank">Play</a>' if recording_url else ""

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

    logger.info("LOG_POINT_130: Parsed %d leads from CSV", len(leads))

    results = []
    for name, phone in leads:
        try:
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

            result = exotel_outbound_call(phone)
            call_sid = result.get("Call", {}).get("Sid") if isinstance(result, dict) else None

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
            logger.exception("LOG_POINT_131: Error handling lead %s (%s)", name, phone)
            results.append({"name": name, "phone": phone, "status": "error", "error": str(e)})

    return JSONResponse({"status": "ok", "results": results})


# ---------------- Realtime media bridge (Exotel <-> OpenAI via MCP-backed tool) ----------------
@app.websocket("/exotel-media")
async def exotel_media_ws(ws: WebSocket):
    """
    Exotel <-> OpenAI Realtime bridge for outbound LIC agent (Shashinath Thakur).

    - Streams caller audio to OpenAI Realtime (gpt-4o-realtime-preview).
    - Streams model audio back to Exotel.
    - At call end, the model calls the `save_call_summary` function tool.
    - We intercept that tool call and forward it to the MCP server's /test-save HTTP endpoint.
    """
    await ws.accept()
    logger.info("LOG_POINT_010: Exotel WS connected & accepted (LIC agent realtime)")

    # --- Call metadata (per stream) ---
    call_id: Optional[str] = None        # Exotel CallSid
    caller_number: Optional[str] = None  # customer phone
    stream_sid: Optional[str] = None     # Exotel stream ID
    # -----------------------------------

    if not OPENAI_API_KEY:
        logger.error("LOG_POINT_011: No OPENAI_API_KEY; closing Exotel stream.")
        await ws.close()
        return

    logger.info("LOG_POINT_012: Using realtime model: %s", REALTIME_MODEL)

    # Exotel stream sequence/timing
    seq_num = 1
    chunk_num = 1
    start_ts = time.time()

    # OpenAI Realtime session
    openai_session: Optional[ClientSession] = None
    openai_ws = None
    pump_task: Optional[asyncio.Task] = None

    # Flags for first occurrences
    first_media_from_exotel = True
    first_audio_to_exotel = True

    # For tool-call argument streaming
    tool_calls: Dict[str, Dict[str, Any]] = {}

    async def send_openai(payload: dict):
        nonlocal openai_ws
        if not openai_ws or openai_ws.closed:
            logger.warning("LOG_POINT_013: Cannot send to OpenAI: WS not ready, payload_type=%s", payload.get("type"))
            return
        t = payload.get("type")
        logger.debug("→ OpenAI SEND: %s", t)
        await openai_ws.send_json(payload)

    async def send_audio_to_exotel(pcm8: bytes):
        """
        Send 8kHz PCM16 audio back to Exotel as media frames.
        Uses the current stream_sid and sequence counters.
        """
        nonlocal seq_num, chunk_num, start_ts, stream_sid, first_audio_to_exotel

        if not stream_sid:
            logger.warning("LOG_POINT_014: No stream_sid; cannot send audio to Exotel yet")
            return

        FRAME_BYTES = 320  # 20 ms at 8kHz mono 16-bit
        now_ms = lambda: int((time.time() - start_ts) * 1000)

        for i in range(0, len(pcm8), FRAME_BYTES):
            chunk_bytes = pcm8[i: i + FRAME_BYTES]
            if not chunk_bytes:
                continue

            if first_audio_to_exotel:
                logger.info("LOG_POINT_070: First audio chunk from OpenAI being sent to Exotel")
                first_audio_to_exotel = False

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

    async def handle_tool_call(tool_name: str, args: Dict[str, Any]):
        """
        Bridge Realtime function tool call -> MCP HTTP endpoint (/test-save).
        """
        logger.info("LOG_POINT_200: Handling tool call: %s args=%s", tool_name, args)
        if tool_name == "save_call_summary":
            if not LIC_CRM_MCP_BASE_URL:
                logger.warning("LOG_POINT_201: LIC_CRM_MCP_BASE_URL not set; cannot forward save_call_summary")
                return
            url = f"{LIC_CRM_MCP_BASE_URL}/test-save"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=args)
                logger.info(
                    "LOG_POINT_202: save_call_summary forwarded to MCP: status=%s body=%s",
                    resp.status_code,
                    resp.text,
                )
            except Exception as e:
                logger.exception("LOG_POINT_203: Error calling MCP save_call_summary: %s", e)
        else:
            logger.warning("LOG_POINT_204: Unknown tool name from model: %s", tool_name)

    async def connect_openai(conn_call_id: str, conn_caller_number: str):
        nonlocal openai_session, openai_ws, pump_task

        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            }

            url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
            logger.info("LOG_POINT_040: Connecting to OpenAI Realtime WS at %s ...", url)

            openai_session = ClientSession()
            openai_ws = await openai_session.ws_connect(url, headers=headers)
            logger.info("LOG_POINT_041: Connected to OpenAI WS")

            # ---------------- LIC persona + tool usage instructions ----------------
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
                "   then wait silently for the caller to speak.\n"
                "5. Never talk about topics outside LIC insurance and basic financial planning.\n\n"
                "TOOL USAGE (VERY IMPORTANT):\n"
                "- You have a function tool `save_call_summary`.\n"
                "- When the phone call is clearly ending (final goodbye), you MUST:\n"
                "  (a) Infer: intent, interest_score (0–10), next_action, and a 3–6 sentence raw_summary.\n"
                "  (b) Call `save_call_summary` exactly once with:\n"
                "      call_id, phone_number, customer_name (if known), interest_score, intent, next_action, raw_summary.\n"
                "Do NOT call save_call_summary in the middle of the conversation; only once at the end."
            )

            # Define Realtime tools (function calling)
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
                                "description": "Unique call id (e.g. Exotel CallSid).",
                            },
                            "phone_number": {
                                "type": "string",
                                "description": "Customer phone number.",
                            },
                            "customer_name": {
                                "type": "string",
                                "description": "Customer name if known, else empty.",
                            },
                            "interest_score": {
                                "type": "integer",
                                "description": "0–10 score of interest in buying LIC.",
                            },
                            "intent": {
                                "type": "string",
                                "description": "Intent label: buy_term, renew, info_only, not_interested, other.",
                            },
                            "next_action": {
                                "type": "string",
                                "description": "follow_up, whatsapp_quote, no_contact, other.",
                            },
                            "raw_summary": {
                                "type": "string",
                                "description": "3–6 sentence natural language summary of the full call.",
                            },
                        },
                        "required": [
                            "call_id",
                            "phone_number",
                            "interest_score",
                            "intent",
                            "next_action",
                            "raw_summary",
                        ],
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
            logger.info(
                "LOG_POINT_042: Built session_config for realtime (no mcp_servers), model=%s",
                REALTIME_MODEL,
            )

            # Send initial session.update
            await send_openai({"type": "session.update", "session": session_config})
            logger.info("LOG_POINT_050: Sent session.update with LIC persona + tools config to OpenAI")

            # Ask the model to start the first greeting turn
            logger.info("LOG_POINT_060: Sending initial response.create to have model greet caller")
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
                            logger.exception("LOG_POINT_080: Failed to parse OpenAI WS message")
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
                                logger.exception("LOG_POINT_081: Failed to decode audio delta")
                                continue

                            try:
                                pcm8 = downsample_24k_to_8k_pcm16(pcm24)
                            except Exception:
                                logger.exception("LOG_POINT_082: Downsampling 24k -> 8k failed")
                                continue

                            await send_audio_to_exotel(pcm8)

                        # Function call item added
                        elif et == "response.output_item.added":
                            item = evt.get("item") or {}
                            if item.get("type") == "function_call":
                                name = item.get("name")
                                call_id_fc = item.get("call_id") or item.get("id")
                                if name and call_id_fc:
                                    tool_calls[call_id_fc] = {"name": name, "arguments": ""}
                                    logger.info("LOG_POINT_210: Tool call started: %s (%s)", name, call_id_fc)

                        # Streaming function call arguments
                        elif et == "response.function_call_arguments.delta":
                            call_id_fc = evt.get("call_id")
                            delta_args = evt.get("arguments_delta") or ""
                            if call_id_fc and call_id_fc in tool_calls:
                                tool_calls[call_id_fc]["arguments"] += delta_args
                                logger.debug(
                                    "LOG_POINT_211: Accumulating tool args for %s: %s",
                                    call_id_fc,
                                    delta_args,
                                )

                        elif et == "response.function_call_arguments.done":
                            call_id_fc = evt.get("call_id")
                            if call_id_fc and call_id_fc in tool_calls:
                                name = tool_calls[call_id_fc]["name"]
                                arg_str = tool_calls[call_id_fc]["arguments"]
                                logger.info(
                                    "LOG_POINT_212: Tool call done: %s (%s) args=%s",
                                    name,
                                    call_id_fc,
                                    arg_str,
                                )
                                try:
                                    args = json.loads(arg_str or "{}")
                                except Exception:
                                    logger.exception("LOG_POINT_213: Failed to parse tool arguments JSON")
                                    args = {}
                                await handle_tool_call(name, args)
                                tool_calls.pop(call_id_fc, None)

                        elif et in (
                            "response.audio.done",
                            "response.output_audio.done",
                            "response.done",
                        ):
                            logger.info("LOG_POINT_083: OpenAI response finished.")

                        elif et == "error":
                            logger.error("LOG_POINT_084: OpenAI ERROR event: %s", evt)

                except Exception as e:
                    logger.exception("LOG_POINT_085: Pump error: %s", e)

            pump_task = asyncio.create_task(pump())

        except Exception as e:
            logger.exception("LOG_POINT_043: OpenAI connection error: %s", e)

    try:
        logger.info("LOG_POINT_015: Entering Exotel WS main loop")
        openai_started = False

        while True:
            raw = await ws.receive_text()
            evt = json.loads(raw)
            ev = evt.get("event")
            logger.info("LOG_POINT_016: Exotel EVENT received: %s - msg=%s", ev, evt)

            if ev == "connected":
                logger.info("LOG_POINT_017: Exotel 'connected' event (handshake)")
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
                    "LOG_POINT_020: Exotel 'start' received: stream_sid=%s, call_id=%s, from=%s",
                    stream_sid,
                    call_id,
                    caller_number,
                )

                CALL_TRANSCRIPTS[stream_sid] = {
                    "call_id": call_id,
                    "phone_number": caller_number,
                    "turns": [],
                }

                if not openai_started:
                    openai_started = True
                    logger.info("LOG_POINT_021: Calling connect_openai with call_id=%s phone=%s", call_id, caller_number)
                    await connect_openai(call_id or "unknown_call", caller_number or "")

            elif ev == "media":
                # Caller audio (8kHz PCM16) -> upsample to 24kHz -> send to OpenAI
                media = evt.get("media") or {}
                payload_b64 = media.get("payload")
                if first_media_from_exotel:
                    logger.info("LOG_POINT_025: First Exotel 'media' frame received (audio from caller)")
                    first_media_from_exotel = False

                if payload_b64 and openai_ws and not openai_ws.closed:
                    try:
                        pcm8 = base64.b64decode(payload_b64)
                    except Exception:
                        logger.warning("LOG_POINT_026: Invalid base64 in Exotel media payload")
                        continue

                    pcm24 = upsample_8k_to_24k_pcm16(pcm8)
                    audio_b64 = base64.b64encode(pcm24).decode("ascii")
                    await send_openai(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        }
                    )
                    logger.debug("LOG_POINT_027: Sent caller audio chunk to OpenAI input_audio_buffer")
                    # server_vad will auto-commit when end-of-speech is detected

            elif ev == "stop":
                logger.info(
                    "LOG_POINT_030: Exotel sent 'stop'; asking model to summarise and call save_call_summary."
                )

                meta = CALL_TRANSCRIPTS.get(stream_sid) or {}
                meta_call_id = meta.get("call_id") or call_id or (stream_sid or "unknown_call")
                meta_phone = meta.get("phone_number") or caller_number or ""

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
                                        "Summarise the entire call and then invoke the "
                                        "`save_call_summary` function tool exactly once with:\n"
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
                logger.info("LOG_POINT_031: Sent summarisation instructions + response.create to OpenAI")

                if stream_sid in CALL_TRANSCRIPTS:
                    CALL_TRANSCRIPTS.pop(stream_sid, None)

                break

            else:
                logger.warning("LOG_POINT_018: Unhandled Exotel event: %s", ev)

    except WebSocketDisconnect:
        logger.info("LOG_POINT_019: Exotel WS disconnected")
    except Exception as e:
        logger.exception("LOG_POINT_032: Error in exotel_media_ws: %s", e)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ws_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        reload=False,
        workers=1,
    )
