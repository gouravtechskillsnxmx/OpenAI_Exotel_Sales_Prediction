# tools_call_summary.py
from typing import Dict, Any
from sqlalchemy.orm import Session
from db_models import CallSummary

def save_call_summary_tool(db: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool: save_call_summary
    Input payload shape:
    {
      "call_id": "string",
      "phone_number": "string",
      "customer_name": "string or null",
      "interest_score": 0-10,
      "intent": "buy_term | renew | info_only | not_interested | other",
      "next_action": "follow_up | whatsapp_quote | no_contact | other",
      "raw_summary": "free text"
    }
    """
    cs = CallSummary(
        call_id=payload["call_id"],
        phone_number=payload["phone_number"],
        customer_name=payload.get("customer_name"),
        intent=payload["intent"],
        interest_score=int(payload["interest_score"]),
        next_action=payload.get("next_action"),
        raw_summary=payload["raw_summary"],
    )
    db.add(cs)
    db.commit()
    db.refresh(cs)

    return {
        "status": "ok",
        "id": cs.id,
    }
