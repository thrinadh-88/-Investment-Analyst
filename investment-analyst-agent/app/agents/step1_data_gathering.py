"""
Step 1 — Data Gathering
Fetches price history, fundamentals, financial statements, news, and analyst
opinions for every primary + peer ticker and stores them in state.
"""
from __future__ import annotations

import logging

from app.models.state import AnalysisState, NewsItem
from app.services.data_sources import (
    fetch_analyst_opinions,
    fetch_financials,
    fetch_fundamentals,
    fetch_news,
    fetch_price_history,
)

logger = logging.getLogger(__name__)


def run(state: AnalysisState) -> AnalysisState:
    all_tickers = list(dict.fromkeys(state.primary_tickers + state.peer_tickers))
    logger.info("Step 1 | tickers=%s period=%s", all_tickers, state.period)

    for ticker in all_tickers:
        logger.debug("  fetching %s …", ticker)

        # Price history
        df = fetch_price_history(ticker, state.period)
        state.price_history[ticker] = df.reset_index().to_dict(orient="records")

        # Fundamentals
        state.fundamentals[ticker] = fetch_fundamentals(ticker)

        # Financial statements
        state.financials[ticker] = fetch_financials(ticker)

        # Analyst opinions
        state.analyst_opinions[ticker] = fetch_analyst_opinions(ticker)

    # News — only for primary tickers (avoid duplication for peers)
    all_news: list[NewsItem] = []
    seen_titles: set[str] = set()
    for ticker in state.primary_tickers:
        raw_items = fetch_news(ticker)
        for item in raw_items:
            title = item.get("title", "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                # Build NewsItem safely — only pass known string fields, skip confidence/sentiment
                all_news.append(NewsItem(
                    title     = item.get("title", ""),
                    summary   = item.get("summary", ""),
                    published = item.get("published", ""),
                    source    = item.get("source", ""),
                    url       = item.get("url", ""),
                ))
    state.news_items = all_news

    logger.info(
        "Step 1 done | tickers=%d news=%d",
        len(all_tickers),
        len(all_news),
    )
    return state
