# summarise_and_save.py
import json
from typing import List, Tuple
from openai import OpenAI
from db_session import SessionLocal
from tools_call_summary import save_call_summary_tool

client = OpenAI()  # assumes OPENAI_API_KEY env var

# 1) Define the tool schema for the model
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_call_summary",
            "description": (
                "Save a structured summary of the LIC sales call into the database. "
                "Use this AFTER the call ends. "
                "Interest_score is 0 (never buying) to 10 (very likely to buy soon)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "call_id": {"type": "string"},
                    "phone_number": {"type": "string"},
                    "customer_name": {"type": "string"},
                    "interest_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "intent": {
                        "type": "string",
                        "description": "e.g. buy_term, renew, info_only, not_interested, other"
                    },
                    "next_action": {
                        "type": "string",
                        "description": "e.g. follow_up, whatsapp_quote, no_contact, other"
                    },
                    "raw_summary": {
                        "type": "string",
                        "description": "Natural language summary of the call in 3–6 sentences."
                    }
                },
                "required": ["call_id", "phone_number", "interest_score", "intent", "raw_summary"]
            }
        }
    }
]

SYSTEM_PROMPT = """
You are Mr. Shashinath Thakur, a senior LIC sales analyst.
You receive the FULL transcript of a call between you (LIC agent) and a customer.
Your job is to:
1) Understand what the customer really wants, their objections, and how interested they are.
2) Assign an interest_score between 0 and 10:
   - 0 = never buying
   - 3–4 = low interest
   - 5–6 = maybe later, some interest
   - 7–8 = good prospect, likely to buy soon
   - 9–10 = very hot lead, highly likely to buy quickly
3) Decide an intent label: buy_term, renew, info_only, not_interested, other.
4) Decide a next_action: follow_up, whatsapp_quote, no_contact, other.
5) Call the tool save_call_summary with a raw_summary of 3–6 sentences.
Do not ask questions, just think and call the tool once.
"""

def build_transcript_text(transcript: List[Tuple[str, str]]) -> str:
    """
    transcript: list of (speaker, text)
    speaker is "agent" or "customer"
    """
    lines = []
    for speaker, text in transcript:
        role = "Agent" if speaker == "agent" else "Customer"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def summarise_and_save_call_summary(call_id: str, phone_number: str,
                                    transcript: List[Tuple[str, str]]) -> dict:
    """
    1) Ask the model to summarise the transcript using the save_call_summary tool.
    2) Execute the tool by writing to DB.
    3) Return whatever the tool returned.
    """
    transcript_text = build_transcript_text(transcript)

    db = SessionLocal()
    try:
        # First call: let the model decide tool args
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",   # choose a cheap but good model
            tools=TOOLS,
            tool_choice="auto",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Call ID: {call_id}\n"
                        f"Phone: {phone_number}\n\n"
                        f"Full transcript:\n{transcript_text}"
                    ),
                },
            ],
        )

        msg = resp.choices[0].message

        if not msg.tool_calls:
            # Model didn't call the tool; fallback: do nothing or log
            return {"status": "no_tool_call"}

        tool_call = msg.tool_calls[0]
        if tool_call.function.name != "save_call_summary":
            return {"status": "unexpected_tool", "tool": tool_call.function.name}

        tool_args = json.loads(tool_call.function.arguments)

        # Ensure required fields from our side are present
        tool_args.setdefault("call_id", call_id)
        tool_args.setdefault("phone_number", phone_number)

        # Execute the tool (write to DB)
        result = save_call_summary_tool(db, tool_args)
        return {"status": "ok", "tool_result": result}

    finally:
        db.close()
