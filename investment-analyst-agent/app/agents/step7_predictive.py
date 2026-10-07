"""
Step 7 — Predictive Insights & Risk Detection
Computes moving averages, momentum, linear price projection, and
five risk indicators (leverage, margin compression, volatility,
sentiment spike, valuation) each rated Low / Medium / High.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.models.state import AnalysisState, RiskIndicator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(score: float) -> str:
    if score >= 0.67:
        return "High"
    if score >= 0.34:
        return "Medium"
    return "Low"


def _prices(state: AnalysisState, ticker: str) -> pd.Series | None:
    records = state.price_history.get(ticker)
    if not records:
        return None
    df = pd.DataFrame(records)
    if "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df.get("Date", df.index))
    return df.set_index("Date").sort_index()["Close"].astype(float)


def _moving_averages(prices: pd.Series) -> dict[str, Any]:
    ma20 = prices.rolling(20).mean().iloc[-1]
    ma50 = prices.rolling(50).mean().iloc[-1]
    last = prices.iloc[-1]
    return {
        "ma20": round(float(ma20), 2) if not np.isnan(ma20) else None,
        "ma50": round(float(ma50), 2) if not np.isnan(ma50) else None,
        "last_price": round(float(last), 2),
        "above_ma20": bool(last > ma20) if not np.isnan(ma20) else None,
        "above_ma50": bool(last > ma50) if not np.isnan(ma50) else None,
        "ma_signal":  "bullish" if (not np.isnan(ma20) and not np.isnan(ma50) and ma20 > ma50) else "bearish",
    }


def _linear_forecast(prices: pd.Series, days: int = 30) -> dict[str, Any]:
    """Simple linear regression forecast."""
    n    = len(prices)
    x    = np.arange(n)
    y    = prices.values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 10:
        return {}
    x, y = x[mask], y[mask]
    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    forecast_price = coef[0] * (n + days) + coef[1]
    pct_change = (forecast_price - y[-1]) / (y[-1] + 1e-9)
    return {
        "forecast_30d": round(float(forecast_price), 2),
        "pct_change_30d": round(float(pct_change), 4),
        "trend_slope": round(float(slope), 4),
        "trend_direction": "up" if slope > 0 else "down",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Risk scoring
# ─────────────────────────────────────────────────────────────────────────────

def _leverage_risk(state: AnalysisState, ticker: str) -> RiskIndicator:
    r = state.ratios.get(ticker)
    de = r.debt_to_equity if r else None
    if de is None:
        score, expl = 0.4, "Debt-to-equity data unavailable."
    elif de > 150:
        score, expl = 0.85, f"High leverage: D/E={de:.1f} (>150)."
    elif de > 80:
        score, expl = 0.55, f"Moderate leverage: D/E={de:.1f}."
    else:
        score, expl = 0.15, f"Low leverage: D/E={de:.1f}."
    return RiskIndicator(name="Leverage Risk", level=_level(score), score=round(score, 2), explanation=expl)


def _margin_compression_risk(state: AnalysisState, ticker: str) -> RiskIndicator:
    k   = state.kpis.get(ticker)
    nm  = k.net_margin        if k else None
    yoy = k.net_income_yoy    if k else None
    if nm is None:
        score, expl = 0.4, "Net margin data unavailable."
    elif nm < 0.05:
        score, expl = 0.80, f"Very thin net margin: {nm*100:.1f}%."
    elif nm < 0.12 and yoy is not None and yoy < -0.05:
        score, expl = 0.60, f"Declining margin: {nm*100:.1f}% with YoY drop of {yoy*100:.1f}%."
    elif nm >= 0.20:
        score, expl = 0.10, f"Healthy net margin: {nm*100:.1f}%."
    else:
        score, expl = 0.30, f"Adequate net margin: {nm*100:.1f}%."
    return RiskIndicator(name="Margin Compression", level=_level(score), score=round(score, 2), explanation=expl)


def _volatility_risk(state: AnalysisState, ticker: str) -> RiskIndicator:
    r   = state.ratios.get(ticker)
    vol = r.volatility_ann if r else None
    if vol is None:
        score, expl = 0.4, "Volatility data unavailable."
    elif vol > 0.45:
        score, expl = 0.85, f"Very high annualised volatility: {vol*100:.1f}%."
    elif vol > 0.25:
        score, expl = 0.50, f"Moderate volatility: {vol*100:.1f}%."
    else:
        score, expl = 0.15, f"Low volatility: {vol*100:.1f}%."
    return RiskIndicator(name="Volatility Risk", level=_level(score), score=round(score, 2), explanation=expl)


def _sentiment_risk(state: AnalysisState) -> RiskIndicator:
    s = state.overall_sentiment
    if s < -0.4:
        score, expl = 0.85, f"Strongly negative market sentiment ({s:.2f})."
    elif s < -0.1:
        score, expl = 0.55, f"Mildly negative market sentiment ({s:.2f})."
    elif s > 0.3:
        score, expl = 0.10, f"Positive market sentiment ({s:.2f})."
    else:
        score, expl = 0.30, f"Neutral market sentiment ({s:.2f})."
    return RiskIndicator(name="Sentiment Risk", level=_level(score), score=round(score, 2), explanation=expl)


def _valuation_risk(state: AnalysisState, ticker: str) -> RiskIndicator:
    r  = state.ratios.get(ticker)
    pe = r.pe if r else None
    pb = r.pb if r else None
    sector_pe = state.sector_median.get("pe")
    if pe is None:
        score, expl = 0.4, "P/E data unavailable."
    elif sector_pe and pe > sector_pe * 1.5:
        score, expl = 0.75, f"P/E={pe:.1f} is >50% above sector median ({sector_pe:.1f})."
    elif pe > 35:
        score, expl = 0.65, f"Elevated P/E: {pe:.1f}."
    elif pe < 10:
        score, expl = 0.20, f"Potentially undervalued: P/E={pe:.1f}."
    else:
        score, expl = 0.25, f"Reasonable valuation: P/E={pe:.1f}."
    return RiskIndicator(name="Valuation Risk", level=_level(score), score=round(score, 2), explanation=expl)


# ─────────────────────────────────────────────────────────────────────────────
# Main step function
# ─────────────────────────────────────────────────────────────────────────────

def run(state: AnalysisState) -> AnalysisState:
    primary = state.primary_tickers
    logger.info("Step 7 | predictive insights for %d primary tickers", len(primary))

    trend_signals: dict[str, Any]  = {}
    price_forecast: dict[str, Any] = {}

    for ticker in primary:
        prices = _prices(state, ticker)
        if prices is not None and len(prices) >= 20:
            trend_signals[ticker]  = _moving_averages(prices)
            price_forecast[ticker] = _linear_forecast(prices)
        else:
            trend_signals[ticker]  = {}
            price_forecast[ticker] = {}
            logger.warning("  %s — insufficient price data for trend analysis.", ticker)

    state.trend_signals  = trend_signals
    state.price_forecast = price_forecast

    # ── Risk indicators (aggregate across all primary tickers) ───────────────
    # Use the first primary ticker for single-ticker risks; sentiment is global
    anchor = primary[0] if primary else ""
    risks: list[RiskIndicator] = [
        _leverage_risk(state, anchor),
        _margin_compression_risk(state, anchor),
        _volatility_risk(state, anchor),
        _sentiment_risk(state),
        _valuation_risk(state, anchor),
    ]
    state.risk_indicators = risks

    high_risks = [r.name for r in risks if r.level == "High"]
    logger.info(
        "Step 7 done | trend_tickers=%d high_risks=%s",
        len([t for t in primary if trend_signals.get(t)]),
        high_risks,
    )
    return state
