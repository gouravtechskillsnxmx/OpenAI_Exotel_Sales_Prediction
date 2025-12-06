"""
ml_rank_call_logs.py

Rank phone numbers in call_logs from most to least interested in buying
an insurance policy, based on:
  - call summary text
  - call duration embedded in the summary

Usage:
    python ml_rank_call_logs.py --db /data/call_logs.db --top_k 30

Adjust DB path and top_k as needed.
"""

import argparse
import sqlite3
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=str,
        default="call_logs.db",
        help="Path to SQLite database file (call_logs.db)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="How many top customers to show",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ranked_customers.csv",
        help="Where to save the ranked list as CSV",
    )
    return parser.parse_args()


def load_call_logs(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT id, call_id, phone_number, status, summary, created_at FROM call_logs",
            conn,
        )
    finally:
        conn.close()
    return df


def extract_duration_seconds(summary: str) -> int:
    """
    Parse duration from text like:
        'Total call duration: 1m 23s.'
        'Total call duration: 43s.'
    Returns duration in seconds, or 0 if not found.
    """
    if not isinstance(summary, str):
        return 0

    # First try minutes + seconds
    m = re.search(r"Total call duration:\s*(\d+)m\s*(\d+)s", summary)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        return minutes * 60 + seconds

    # Then try only seconds
    m2 = re.search(r"Total call duration:\s*(\d+)s", summary)
    if m2:
        return int(m2.group(1))

    return 0


def build_interest_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure we have a summary column
    df["summary"] = df["summary"].fillna("")

    # 1) Duration in seconds
    df["duration_sec"] = df["summary"].apply(extract_duration_seconds)

    # 2) TF-IDF vectorization of summaries
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(df["summary"])

    # Text interest = norm of TF-IDF vector
    text_interest = np.linalg.norm(X.toarray(), axis=1)

    # Normalize text_interest
    ti_arr = np.array(text_interest, dtype=float)
    max_ti = float(ti_arr.max()) if len(ti_arr) > 0 else 0.0
    if max_ti > 0:
        ti_norm = ti_arr / max_ti
    else:
        ti_norm = np.zeros_like(ti_arr)

    # Normalize duration
    dur_arr = df["duration_sec"].values.astype(float)
    max_dur = float(dur_arr.max()) if len(dur_arr) > 0 else 0.0
    if max_dur > 0:
        dur_norm = dur_arr / max_dur
    else:
        dur_norm = np.zeros_like(dur_arr)

    # 3) Simple keyword intent score
    positive_keywords = [
        "policy",
        "policies",
        "premium",
        "premiums",
        "coverage",
        "retirement",
        "investment",
        "invest",
        "term plan",
        "pension",
        "ulip",
        "bonus",
        "sum assured",
        # Hindi / Hinglish hints
        "policy lene",
        "policy kharidna",
        "policy kharid",
        "interest dikhaya",
        "रुचि",
        "निवेश",
    ]

    def keyword_score(text: str) -> int:
        t = text.lower()
        return sum(1 for kw in positive_keywords if kw in t)

    kw_scores = np.array([keyword_score(s) for s in df["summary"]], dtype=float)
    max_kw = float(kw_scores.max()) if len(kw_scores) > 0 else 0.0
    if max_kw > 0:
        kw_norm = kw_scores / max_kw
    else:
        kw_norm = np.zeros_like(kw_scores)

    # Final interest_score = weighted combination
    interest_score = 0.5 * ti_norm + 0.3 * dur_norm + 0.2 * kw_norm

    df["text_interest"] = ti_norm
    df["duration_norm"] = dur_norm
    df["keyword_norm"] = kw_norm
    df["interest_score"] = interest_score

    return df


def rank_customers(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    # Group by phone_number: keep the row with highest interest_score per phone
    grouped = df.sort_values("interest_score", ascending=False).groupby("phone_number")
    best_rows = grouped.head(1).reset_index(drop=True)

    # Sort again by interest_score desc
    best_rows = best_rows.sort_values("interest_score", ascending=False)

    # Take top_k
    top = best_rows.head(top_k).copy()
    return top


def main():
    args = parse_args()

    print(f"Loading call_logs from: {args.db}")
    df = load_call_logs(args.db)
    if df.empty:
        print("No rows found in call_logs table.")
        return

    print(f"Loaded {len(df)} call_logs rows.")
    print("Building interest scores...")
    df_scored = build_interest_scores(df)

    print("Ranking customers (phone numbers)...")
    top_customers = rank_customers(df_scored, args.top_k)

    # Show in console
    print("\nTOP CUSTOMERS BY INTEREST SCORE:")
    for _, row in top_customers.iterrows():
        summary_clean = row['summary'][:80].replace("\n", " ")  # <- PRECOMPUTED FIX
        print(
            f"{row['phone_number']} | score={row['interest_score']:.4f} | "
            f"duration={row['duration_sec']}s | status={row['status']} | "
            f"summary={summary_clean}..."
        )


    # Save to CSV
    top_customers.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\nSaved ranked customers to: {args.output_csv}")


if __name__ == "__main__":
    main()
