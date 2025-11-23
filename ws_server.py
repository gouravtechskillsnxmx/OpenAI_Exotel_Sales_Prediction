"""
ws_server.py — Exotel Outbound Realtime LIC Agent + Call Logger + MCP tool-calls

Environment variables expected:

  PORT=10000 (Render)
  LOG_LEVEL=INFO or DEBUG

  EXOTEL_SID       gouravnxmx1
  EXOTEL_TOKEN     your token
  EXO_SUBDOMAIN    api or api.in
  EXO_CALLER_ID    your Exophone, e.g. 08047362093

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
EXO_SUBDOMAIN = os.getenv("EXO_SUBDOMAIN", "api")  # "api" or "api.in"
EXO_CALLER_ID = os.getenv("EXO_CALLER_ID", "")

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


# ---------------------------------------------------------
# HTML test page (optional)
# ---------------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
  <head>
    <title>Exotel LIC Voicebot</title>
  </head>
  <body>
    <h1>Exotel LIC Voicebot Backend</h1>
    <p>This service powers Exotel outbound calls + Realtime OpenAI voice agent.</p>
    <ul>
      <li><code>GET /</code> – this page</li>
      <li><code>GET /exotel-ws-bootstrap</code> – used by Exotel Voicebot "Dynamic WS URL"</li>
      <li><code>GET /call_logs</code> – view recent call logs</li>
      <li><code>POST /exotel-outbound-call</code> – trigger outbound call</li>
      <li><code>POST /exotel-status</code> – Exotel status callback (optional)</li>
    </ul>
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
# Simple lead + call log API (optional, for dashboard)
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

    # Trigger Exotel call
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
    Trigger an outbound call using Exotel API.
    """
    if not EXOTEL_SID or not EXOTEL_TOKEN or not EXO_CALLER_ID:
        logger.error("Exotel credentials or caller ID missing; cannot place outbound call.")
        return {"error": "exotel credentials/caller id missing"}

    exotel_url = f"https://{EXO_SUBDOMAIN}.{EXOTEL_SID}:{EXOTEL_TOKEN}@{EXO_SUBDOMAIN}.exotel.com/v1/Accounts/{EXOTEL_SID}/Calls/connect"

    payload = {
        "From": to_number,
        "To": EXO_CALLER_ID,
        "CallerId": EXO_CALLER_ID,
        "Url": "http://my.exotel.com/Exotel/exoml/start/1075544",
    }

    logger.info("Exotel outbound call payload: %s", payload)

    try:
        import requests

        resp = requests.post(exotel_url, data=payload, timeout=15)
        resp.raise_for_status()
        text = resp.text
        logger.info("Exotel outbound call result: %s", text)
        try:
            # Exotel returns XML; parse minimally
            # We just stash the raw result here for logging
            return {"raw": text}
        except Exception:
            return {"raw": text}
    except Exception as e:
        logger.exception("Error placing Exotel outbound call: %s", e)
        return {"error": str(e)}


@app.post("/exotel-outbound-call")
async def exotel_outbound_call_endpoint(request: Request):
    """
    Simple HTTP endpoint to trigger an outbound Exotel call from JSON:
      { "phone": "8850298070" }
    """
    data = await request.json()
    phone = data.get("phone", "").strip()
    if not phone:
        return JSONResponse({"error": "phone is required"}, status_code=400)

    result = exotel_outbound_call(phone)
    return JSONResponse(result)


# ---------------------------------------------------------
# MCP helper: forward tool-call results to LIC_CRM_MCP_BASE_URL
# ---------------------------------------------------------

async def forward_save_call_summary_to_mcp(payload: Dict[str, Any]) -> None:
    """
    Calls LIC_CRM_MCP_BASE_URL/test-save with a JSON body for saving call summary.
    """
    if not LIC_CRM_MCP_BASE_URL:
        logger.warning("LIC_CRM_MCP_BASE_URL not set; cannot forward save_call_summary")
        return
    url = f"{LIC_CRM_MCP_BASE_URL}/test-save"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            logger.info("Forwarding save_call_summary to MCP: %s", url)
            r = await client.post(url, json=payload)
            logger.info("MCP response: status=%s body=%s", r.status_code, r.text)
    except Exception:
        logger.exception("Error forwarding save_call_summary to MCP server")


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
                "2. Confirm you are calling about LIC policies.\n"
                "3. Ask a few probing questions about their existing insurance, "
                "   family, financial goals, and risk appetite.\n"
                "4. Recommend suitable LIC plans (e.g., term, endowment, ULIP, pension) "
                "   with simple explanation (no jargon).\n"
                "5. Be concise, polite, and not pushy.\n"
                "6. At the end, summarise the conversation: what you understood, "
                "   what you recommended, and any next steps.\n\n"
                "IMPORTANT:\n"
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

                        # Function call item added
                        elif et == "response.output_item.added":
                            item = (evt.get("item") or {}) or {}
                            if item.get("type") == "response.function_call":
                                fc = item.get("function") or {}
                                tool_name = fc.get("name")
                                args = fc.get("arguments") or {}
                                logger.info(
                                    "Tool-call received: name=%s args=%s", tool_name, args
                                )

                                if tool_name == "save_call_summary":
                                    call_id_param = args.get("call_id") or conn_call_id
                                    phone_param = args.get("phone_number") or conn_caller_number
                                    summary_param = args.get("summary") or ""

                                    # Persist into local DB + forward to MCP
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

                                    await forward_save_call_summary_to_mcp(
                                        {
                                            "call_id": call_id_param,
                                            "phone_number": phone_param,
                                            "summary": summary_param,
                                        }
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
                start_obj = evt.get("start") or {}
                stream_sid = start_obj.get("stream_sid") or evt.get("stream_sid")
                start_ts = time.time()

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
                logger.info("Exotel sent stop; closing WS and letting model wrap up.")

                # Fetch metadata
                meta = CALL_TRANSCRIPTS.get(stream_sid) or {}
                meta_call_id = meta.get("call_id") or call_id or (stream_sid or "unknown_call")
                meta_phone = meta.get("phone_number") or caller_number or ""

                # Save minimal record if we don't already have one
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO call_logs (call_id, phone_number, status, summary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (meta_call_id, meta_phone, "stopped", ""),
                )
                conn.commit()
                conn.close()

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
