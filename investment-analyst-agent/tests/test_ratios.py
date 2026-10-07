"""
tests/test_ratios.py — unit tests for financial ratio calculations.
Uses a synthetic state populated with known values to verify math.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.models.state import AnalysisState
from app.agents import step4_ratios


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_state_with_prices(ticker: str = "TEST", prices: list[float] | None = None) -> AnalysisState:
    if prices is None:
        np.random.seed(42)
        prices = list(100 * np.cumprod(1 + np.random.normal(0, 0.01, 252)))

    dates = pd.date_range("2023-01-01", periods=len(prices), freq="B")
    records = [
        {"Date": str(d.date()), "Close": p, "Open": p, "High": p * 1.01, "Low": p * 0.99, "Volume": 1000000}
        for d, p in zip(dates, prices)
    ]
    state = AnalysisState(
        primary_tickers=[ticker],
        peer_tickers=[],
        mock_mode=True,
    )
    state.price_history[ticker] = records
    state.fundamentals[ticker] = {
        "trailingPE":       25.0,
        "priceToBook":       4.5,
        "returnOnEquity":    0.22,
        "returnOnAssets":    0.10,
        "debtToEquity":      45.0,
        "currentRatio":       2.1,
        "interestCoverage":   8.5,
        "beta":               1.1,
        "currentPrice":     prices[-1],
        "trailingEps":      prices[-1] / 25.0,
    }
    return state


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pe_ratio_from_fundamentals():
    state = _make_state_with_prices()
    result = step4_ratios.run(state)
    assert result.ratios["TEST"].pe == 25.0


def test_pb_ratio_from_fundamentals():
    state = _make_state_with_prices()
    result = step4_ratios.run(state)
    assert result.ratios["TEST"].pb == 4.5


def test_roe_from_fundamentals():
    state = _make_state_with_prices()
    result = step4_ratios.run(state)
    assert result.ratios["TEST"].roe == pytest.approx(0.22, abs=1e-6)


def test_volatility_is_positive():
    state = _make_state_with_prices()
    result = step4_ratios.run(state)
    vol = result.ratios["TEST"].volatility_ann
    assert vol is not None
    assert vol > 0.0


def test_max_drawdown_is_negative_or_zero():
    state = _make_state_with_prices()
    result = step4_ratios.run(state)
    dd = result.ratios["TEST"].max_drawdown
    assert dd is not None
    assert dd <= 0.0


def test_max_drawdown_known_values():
    """Manually construct a price series with known max drawdown of ~-50%."""
    prices = [100, 120, 80, 60, 90, 110]  # peak=120, trough=60 → dd = (60-120)/120 = -0.5
    state  = _make_state_with_prices(prices=prices)
    result = step4_ratios.run(state)
    dd = result.ratios["TEST"].max_drawdown
    assert dd == pytest.approx(-0.5, abs=0.02)


def test_annualised_volatility_flat_prices():
    """Flat prices → zero volatility."""
    prices = [100.0] * 100
    state  = _make_state_with_prices(prices=prices)
    result = step4_ratios.run(state)
    vol = result.ratios["TEST"].volatility_ann
    assert vol == pytest.approx(0.0, abs=1e-6)


def test_debt_to_equity_from_fundamentals():
    state = _make_state_with_prices()
    result = step4_ratios.run(state)
    assert result.ratios["TEST"].debt_to_equity == 45.0


def test_beta_computed_from_prices():
    """When fundamentals don't provide beta, it should be computed from prices.
    When ticker IS the market proxy, beta approaches 1.0."""
    np.random.seed(0)
    prices = list(100 * np.cumprod(1 + np.random.normal(0, 0.01, 300)))
    state  = _make_state_with_prices(prices=prices)
    # Remove beta from fundamentals to force price-based computation
    state.fundamentals["TEST"].pop("beta", None)
    result = step4_ratios.run(state)
    beta = result.ratios["TEST"].beta
    assert beta is not None
    # When ticker IS the market proxy, beta should be exactly 1.0
    assert beta == pytest.approx(1.0, abs=1e-9)
