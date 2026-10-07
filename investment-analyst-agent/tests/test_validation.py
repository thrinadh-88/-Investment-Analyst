"""
tests/test_validation.py — unit tests for Step 2 Validation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.models.state import AnalysisState, NewsItem
from app.agents import step2_validation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prices(n: int = 30, stale: bool = False) -> list[dict]:
    if stale:
        end = datetime.now(timezone.utc) - timedelta(days=10)
    else:
        end = datetime.now(timezone.utc)
    dates = pd.date_range(end=end, periods=n, freq="B")
    return [
        {"Date": str(d), "Close": float(100 + i), "Open": 100.0, "High": 102.0, "Low": 98.0, "Volume": 1e6}
        for i, d in enumerate(dates)
    ]


def _base_state(ticker: str = "AAPL") -> AnalysisState:
    state = AnalysisState(primary_tickers=[ticker], peer_tickers=[], mock_mode=True)
    state.price_history[ticker] = _make_prices()
    state.fundamentals[ticker] = {
        "trailingPE": 25.0, "priceToBook": 4.0, "returnOnEquity": 0.20,
        "totalRevenue": 1e10, "netIncomeToCommon": 2e9, "profitMargins": 0.20,
        "currency": "USD",
    }
    return state


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_quality_score_perfect_data():
    state  = _base_state()
    result = step2_validation.run(state)
    # Perfect data should score close to 100 (no news → -5 deduction)
    assert result.data_quality_score >= 90.0


def test_missing_price_data_flag():
    state = _base_state()
    state.price_history["AAPL"] = []   # empty
    result = step2_validation.run(state)
    assert any("missing_price_data" in f for f in result.validation_flags)
    assert result.data_quality_score < 100.0


def test_stale_price_data_flag():
    state = _base_state()
    state.price_history["AAPL"] = _make_prices(stale=True)
    result = step2_validation.run(state)
    assert any("stale_price_data" in f for f in result.validation_flags)


def test_missing_fundamentals_flag():
    state = _base_state()
    state.fundamentals["AAPL"] = {}   # empty
    result = step2_validation.run(state)
    assert any("missing_fundamentals" in f for f in result.validation_flags)


def test_no_news_flag():
    state = _base_state()
    state.news_items = []
    result = step2_validation.run(state)
    assert any("no_news" in f for f in result.validation_flags)


def test_duplicate_news_deduplication():
    state = _base_state()
    state.news_items = [
        NewsItem(title="Stock rises on earnings", published="2024-01-01"),
        NewsItem(title="Stock rises on earnings", published="2024-01-01"),  # duplicate
        NewsItem(title="Different headline",      published="2024-01-02"),
    ]
    result = step2_validation.run(state)
    assert len(result.news_items) == 2
    assert any("duplicate_news_removed" in f for f in result.validation_flags)


def test_currency_mismatch_flag():
    state = _base_state()
    state.fundamentals["AAPL"]["currency"] = "USD"
    state.peer_tickers = ["TCS"]
    state.price_history["TCS"] = _make_prices()
    state.fundamentals["TCS"] = {
        "trailingPE": 20.0, "priceToBook": 3.0, "returnOnEquity": 0.18,
        "totalRevenue": 5e9, "netIncomeToCommon": 1e9, "profitMargins": 0.18,
        "currency": "INR",
    }
    result = step2_validation.run(state)
    assert any("currency_mismatch" in f for f in result.validation_flags)


def test_quality_score_range():
    """Quality score must always be 0–100."""
    state = _base_state()
    state.price_history["AAPL"] = []
    state.fundamentals["AAPL"] = {}
    state.news_items = []
    result = step2_validation.run(state)
    assert 0.0 <= result.data_quality_score <= 100.0
