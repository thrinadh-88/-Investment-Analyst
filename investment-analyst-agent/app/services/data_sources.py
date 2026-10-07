"""
app/services/data_sources.py — yfinance + news fetch helpers with mock fallbacks.
All public functions accept a ticker string and return plain Python objects
(DataFrames, dicts, lists) that the pipeline steps consume.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mock data factories
# ─────────────────────────────────────────────────────────────────────────────

def _mock_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 252, "2y": 504}.get(period, 252)
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    np.random.seed(abs(hash(ticker)) % (2**31))
    base = random.uniform(200, 3000)
    prices = base * np.cumprod(1 + np.random.normal(0.0003, 0.015, days))
    df = pd.DataFrame({
        "Open":   prices * np.random.uniform(0.99, 1.00, days),
        "High":   prices * np.random.uniform(1.00, 1.02, days),
        "Low":    prices * np.random.uniform(0.98, 1.00, days),
        "Close":  prices,
        "Volume": np.random.randint(500_000, 5_000_000, days).astype(float),
    }, index=dates)
    df.index.name = "Date"
    return df


def _mock_fundamentals(ticker: str) -> dict[str, Any]:
    np.random.seed(abs(hash(ticker)) % (2**31))
    base_price = random.uniform(200, 3000)
    return {
        "symbol":               ticker,
        "shortName":            f"{ticker} Corp",
        "sector":               "Technology",
        "industry":             "Software",
        "marketCap":            int(base_price * 1_000_000 * random.uniform(50, 500)),
        "trailingPE":           round(random.uniform(12, 45), 2),
        "priceToBook":          round(random.uniform(1.5, 8.0), 2),
        "returnOnEquity":       round(random.uniform(0.08, 0.35), 4),
        "returnOnAssets":       round(random.uniform(0.04, 0.18), 4),
        "debtToEquity":         round(random.uniform(10, 120), 2),
        "currentRatio":         round(random.uniform(1.2, 3.5), 2),
        "beta":                 round(random.uniform(0.6, 1.6), 2),
        "totalRevenue":         int(random.uniform(5e9, 50e9)),
        "netIncomeToCommon":    int(random.uniform(5e8, 8e9)),
        "ebitda":               int(random.uniform(1e9, 12e9)),
        "operatingMargins":     round(random.uniform(0.10, 0.35), 4),
        "profitMargins":        round(random.uniform(0.06, 0.25), 4),
        "trailingEps":          round(random.uniform(5, 80), 2),
        "totalDebt":            int(random.uniform(1e9, 20e9)),
        "freeCashflow":         int(random.uniform(2e8, 6e9)),
        "revenueGrowth":        round(random.uniform(-0.05, 0.25), 4),
        "earningsGrowth":       round(random.uniform(-0.10, 0.30), 4),
        "interestCoverage":     round(random.uniform(3, 20), 2),
        "currency":             "INR" if ticker.endswith(".NS") else "USD",
        "fiftyTwoWeekHigh":     round(base_price * 1.3, 2),
        "fiftyTwoWeekLow":      round(base_price * 0.7, 2),
        "currentPrice":         round(base_price, 2),
        "recommendationKey":    random.choice(["buy", "hold", "underperform"]),
        "numberOfAnalystOpinions": random.randint(5, 30),
    }


def _mock_news(ticker: str) -> list[dict[str, Any]]:
    headlines = [
        f"{ticker} reports strong quarterly earnings beating estimates",
        f"Analysts upgrade {ticker} on improved margin outlook",
        f"{ticker} faces headwinds from global supply chain disruptions",
        f"New product launch drives optimism for {ticker} investors",
        f"{ticker} management provides cautious guidance for next quarter",
        f"Institutional investors increase stake in {ticker}",
        f"{ticker} under pressure amid broader market sell-off",
    ]
    items = []
    for i, h in enumerate(headlines):
        pub_date = datetime.today() - timedelta(days=i * 3)
        items.append({
            "title":     h,
            "summary":   h + ". Analysts are watching developments closely.",
            "published": pub_date.strftime("%Y-%m-%d"),
            "source":    "MockNews",
            "url":       f"https://example.com/news/{ticker.lower()}/{i}",
        })
    return items


def _mock_analyst_opinions(ticker: str) -> dict[str, Any]:
    return {
        "strongBuy":    random.randint(2, 8),
        "buy":          random.randint(3, 10),
        "hold":         random.randint(1, 6),
        "sell":         random.randint(0, 3),
        "strongSell":   random.randint(0, 2),
        "targetMeanPrice": round(random.uniform(200, 4000), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    if settings.mock_mode:
        return _mock_price_history(ticker, period)
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df.empty:
            logger.warning("fetch_price_history: empty data for %s, using mock.", ticker)
            return _mock_price_history(ticker, period)
        return df
    except Exception as exc:
        logger.error("fetch_price_history(%s) failed: %s — using mock.", ticker, exc)
        return _mock_price_history(ticker, period)


def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    if settings.mock_mode:
        return _mock_fundamentals(ticker)
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if not info or len(info) < 5:
            logger.warning("fetch_fundamentals: sparse data for %s, using mock.", ticker)
            return _mock_fundamentals(ticker)
        return info
    except Exception as exc:
        logger.error("fetch_fundamentals(%s) failed: %s — using mock.", ticker, exc)
        return _mock_fundamentals(ticker)


def fetch_financials(ticker: str) -> dict[str, Any]:
    """Returns income statement rows as a dict of {metric: {date: value}}."""
    if settings.mock_mode:
        return {}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        fin = t.financials
        if fin is None or fin.empty:
            return {}
        return fin.to_dict()
    except Exception as exc:
        logger.error("fetch_financials(%s) failed: %s", ticker, exc)
        return {}


def fetch_news(ticker: str) -> list[dict[str, Any]]:
    """Returns list of news item dicts with title/summary/published/source/url."""
    if settings.mock_mode:
        return _mock_news(ticker)

    items: list[dict[str, Any]] = []

    # ── Try NewsAPI ───────────────────────────────────────────────────────────
    if settings.news_api_key:
        try:
            import requests
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        ticker.replace(".NS", "").replace(".BO", ""),
                    "sortBy":   "publishedAt",
                    "pageSize": 15,
                    "language": "en",
                    "apiKey":   settings.news_api_key,
                },
                timeout=10,
            )
            data = resp.json()
            for art in data.get("articles", []):
                items.append({
                    "title":     art.get("title", ""),
                    "summary":   art.get("description", ""),
                    "published": art.get("publishedAt", "")[:10],
                    "source":    art.get("source", {}).get("name", ""),
                    "url":       art.get("url", ""),
                })
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for %s: %s", ticker, exc)

    # ── Fallback: Yahoo Finance RSS ───────────────────────────────────────────
    if not items:
        try:
            import feedparser
            clean = ticker.replace(".NS", "").replace(".BO", "")
            url   = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed  = feedparser.parse(url)
            for entry in feed.entries[:15]:
                items.append({
                    "title":     entry.get("title", ""),
                    "summary":   entry.get("summary", ""),
                    "published": entry.get("published", "")[:10],
                    "source":    "Yahoo Finance",
                    "url":       entry.get("link", ""),
                })
        except Exception as exc:
            logger.warning("RSS feed fetch failed for %s: %s", ticker, exc)

    if not items:
        logger.warning("fetch_news: no news found for %s — using mock.", ticker)
        return _mock_news(ticker)

    return items


def fetch_analyst_opinions(ticker: str) -> dict[str, Any]:
    if settings.mock_mode:
        return _mock_analyst_opinions(ticker)
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        info = t.info or {}
        rec  = t.recommendations
        opinions: dict[str, Any] = {
            "strongBuy":        info.get("numberOfAnalystOpinions", 0),
            "targetMeanPrice":  info.get("targetMeanPrice"),
            "recommendationKey": info.get("recommendationKey", "none"),
        }
        if rec is not None and not rec.empty:
            latest = rec.iloc[-1] if len(rec) else {}
            opinions.update({
                "strongBuy":  int(latest.get("strongBuy",  0)),
                "buy":        int(latest.get("buy",        0)),
                "hold":       int(latest.get("hold",       0)),
                "sell":       int(latest.get("sell",       0)),
                "strongSell": int(latest.get("strongSell", 0)),
            })
        return opinions
    except Exception as exc:
        logger.error("fetch_analyst_opinions(%s) failed: %s", ticker, exc)
        return _mock_analyst_opinions(ticker)
