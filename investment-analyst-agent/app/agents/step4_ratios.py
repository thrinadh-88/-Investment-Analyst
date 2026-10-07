"""
Step 4 — Financial Ratio Calculation
Computes P/E, P/B, ROE, ROA, D/E, current ratio, interest coverage,
beta, annualised volatility, and max drawdown from price history + fundamentals.
All calculations are pure pandas/numpy — no external API calls.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.models.state import AnalysisState, Ratios

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


def _max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown as a negative fraction (e.g. -0.25 = -25%)."""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / (cummax + 1e-9)
    return float(drawdown.min())


def _annualised_volatility(prices: pd.Series) -> float:
    returns = prices.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(_TRADING_DAYS))


def _beta(ticker_prices: pd.Series, market_prices: pd.Series) -> float | None:
    """Rolling beta vs market index using aligned daily returns."""
    try:
        ticker_ret = ticker_prices.pct_change().dropna()
        market_ret = market_prices.pct_change().dropna()
        aligned = pd.concat([ticker_ret, market_ret], axis=1, join="inner")
        aligned.columns = ["t", "m"]
        if len(aligned) < 30:
            return None
        cov = aligned["t"].cov(aligned["m"])
        var = aligned["m"].var()
        if var == 0:
            return None
        return float(cov / var)
    except Exception:
        return None


def _prices_from_state(state: AnalysisState, ticker: str) -> pd.Series | None:
    records = state.price_history.get(ticker)
    if not records:
        return None
    df = pd.DataFrame(records)
    if "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df.get("Date", df.index))
    df = df.set_index("Date").sort_index()
    return df["Close"].astype(float)


def run(state: AnalysisState) -> AnalysisState:
    all_tickers = list(dict.fromkeys(state.primary_tickers + state.peer_tickers))
    logger.info("Step 4 | computing ratios for %d tickers", len(all_tickers))

    # We use the first primary ticker as a rough "market" for beta if no index available
    market_series: pd.Series | None = None
    if state.primary_tickers:
        market_series = _prices_from_state(state, state.primary_tickers[0])

    for ticker in all_tickers:
        fund   = state.fundamentals.get(ticker, {})
        prices = _prices_from_state(state, ticker)

        # ── Use pre-computed ratios from fundamentals where available ─────────
        pe  = fund.get("trailingPE")
        pb  = fund.get("priceToBook")
        roe = fund.get("returnOnEquity")
        roa = fund.get("returnOnAssets")
        de  = fund.get("debtToEquity")
        cr  = fund.get("currentRatio")
        ic  = fund.get("interestCoverage")
        beta_val = fund.get("beta")

        # ── Price-based calculations ──────────────────────────────────────────
        vol       = _annualised_volatility(prices) if prices is not None else None
        max_dd    = _max_drawdown(prices) if prices is not None else None

        # Beta from prices if not in fundamentals
        if beta_val is None and prices is not None and market_series is not None:
            beta_val = _beta(prices, market_series)

        # ── Fallback P/E from price + EPS ─────────────────────────────────────
        if pe is None:
            eps = fund.get("trailingEps") or (state.kpis.get(ticker) and state.kpis[ticker].eps)
            cur_price = fund.get("currentPrice")
            if eps and cur_price and eps != 0:
                pe = round(cur_price / eps, 2)

        # ── Fallback D/E from debt + equity ───────────────────────────────────
        if de is None:
            total_debt   = fund.get("totalDebt")
            market_cap   = fund.get("marketCap")
            if total_debt and market_cap and market_cap > 0:
                de = round(total_debt / market_cap * 100, 2)

        state.ratios[ticker] = Ratios(
            pe               = round(pe, 2)        if pe is not None else None,
            pb               = round(pb, 2)        if pb is not None else None,
            roe              = round(roe, 4)       if roe is not None else None,
            roa              = round(roa, 4)       if roa is not None else None,
            debt_to_equity   = round(de, 2)        if de is not None else None,
            current_ratio    = round(cr, 2)        if cr is not None else None,
            interest_coverage = round(ic, 2)       if ic is not None else None,
            beta             = round(beta_val, 3)  if beta_val is not None else None,
            volatility_ann   = round(vol, 4)       if vol is not None else None,
            max_drawdown     = round(max_dd, 4)    if max_dd is not None else None,
        )
        logger.debug("  %s ratios computed", ticker)

    logger.info("Step 4 done | %d ratio sets", len(state.ratios))
    return state
