"""
Step 2 — Validation
Checks data quality across all tickers: null values, staleness, outliers,
currency mismatches, duplicate news. Assigns a data-quality score (0–100)
and populates state.validation_flags.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.models.state import AnalysisState

logger = logging.getLogger(__name__)

_REQUIRED_FUNDAMENTALS = [
    "trailingPE", "priceToBook", "returnOnEquity", "totalRevenue",
    "netIncomeToCommon", "profitMargins",
]
_STALE_THRESHOLD_DAYS = 4


def _score_penalties(flags: list[str]) -> float:
    """Each flag reduces score by a fixed amount (min 0)."""
    penalty_map = {
        "missing_price_data":      20,
        "stale_price_data":        10,
        "missing_fundamentals":    15,
        "price_outlier_detected":   5,
        "currency_mismatch":        5,
        "no_news":                  5,
        "duplicate_news_removed":   2,
    }
    total_penalty = sum(penalty_map.get(f.split(":")[0].strip(), 3) for f in flags)
    return max(0.0, 100.0 - total_penalty)


def run(state: AnalysisState) -> AnalysisState:
    flags: list[str] = []
    all_tickers = list(dict.fromkeys(state.primary_tickers + state.peer_tickers))
    logger.info("Step 2 | validating %d tickers", len(all_tickers))

    currencies: list[str] = []

    for ticker in all_tickers:
        # ── Price history checks ──────────────────────────────────────────────
        records = state.price_history.get(ticker, [])
        if not records:
            flags.append(f"missing_price_data: {ticker}")
            logger.warning("  %s — no price data", ticker)
            continue

        df = pd.DataFrame(records)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            last_date = df["Date"].max()
            now_utc   = datetime.now(timezone.utc)
            if (now_utc - last_date).days > _STALE_THRESHOLD_DAYS:
                flags.append(f"stale_price_data: {ticker} (last={last_date.date()})")

        # Outlier detection on daily returns (z-score > 3.5)
        if "Close" in df.columns and len(df) > 10:
            closes  = df["Close"].dropna().astype(float)
            returns = closes.pct_change().dropna()
            z_scores = (returns - returns.mean()) / (returns.std() + 1e-9)
            if (z_scores.abs() > 3.5).any():
                flags.append(f"price_outlier_detected: {ticker}")

        # ── Fundamentals checks ───────────────────────────────────────────────
        fund = state.fundamentals.get(ticker, {})
        missing = [k for k in _REQUIRED_FUNDAMENTALS if fund.get(k) is None]
        if len(missing) >= 3:
            flags.append(f"missing_fundamentals: {ticker} ({', '.join(missing[:3])})")

        # Currency tracking
        ccy = fund.get("currency", "")
        if ccy:
            currencies.append(ccy)

    # ── Currency mismatch ─────────────────────────────────────────────────────
    unique_currencies = set(currencies)
    if len(unique_currencies) > 1:
        flags.append(f"currency_mismatch: {sorted(unique_currencies)}")
        logger.warning("  Currency mismatch detected: %s", unique_currencies)

    # ── News checks ───────────────────────────────────────────────────────────
    if not state.news_items:
        flags.append("no_news")
    else:
        # Dedup by title (already done in step 1, but re-verify)
        seen: set[str] = set()
        deduped = []
        removed = 0
        for item in state.news_items:
            key = item.title.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)
            else:
                removed += 1
        if removed:
            flags.append(f"duplicate_news_removed: {removed} items")
            state.news_items = deduped

    state.validation_flags   = flags
    state.data_quality_score = _score_penalties(flags)

    logger.info(
        "Step 2 done | quality_score=%.1f flags=%d",
        state.data_quality_score,
        len(flags),
    )
    return state
