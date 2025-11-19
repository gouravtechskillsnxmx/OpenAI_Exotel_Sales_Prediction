# db_models.py
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CallSummary(Base):
    __tablename__ = "call_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(64), unique=True, index=True, nullable=False)
    phone_number = Column(String(20), index=True, nullable=False)
    customer_name = Column(String(100))
    call_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    intent = Column(String(50), nullable=False)          # e.g. "buy_term", "info_only"
    interest_score = Column(Integer, nullable=False)     # 0–10
    next_action = Column(String(50))                     # e.g. "follow_up", "no_contact"

    raw_summary = Column(Text, nullable=False)           # full natural-language summary

    # label fields for ML later
    purchased = Column(Boolean, default=None)            # filled later when they actually buy
    purchase_date = Column(DateTime, nullable=True)
