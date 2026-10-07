"""
Step 6 — Market Sentiment Analysis
Classifies each news item using Granite (positive / neutral / negative),
builds a sentiment index time-series, and computes overall_sentiment (-1 to +1).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

from app.models.state import AnalysisState, NewsItem
from app.prompts import load_prompt
from app.services.watsonx_client import get_watsonx_client

logger = logging.getLogger(__name__)

_SENTIMENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _classify_item(client, item: NewsItem, template: str) -> NewsItem:
    prompt = template.format(title=item.title, summary=item.summary or item.title)
    result = client.extract_json(prompt, max_tokens=200)
    sentiment  = result.get("sentiment",  "neutral").lower()
    confidence = float(result.get("confidence", 0.5))
    rationale  = result.get("rationale", "")

    if sentiment not in _SENTIMENT_MAP:
        sentiment = "neutral"

    item.sentiment   = sentiment
    item.confidence  = round(confidence, 3)
    item.rationale   = rationale
    return item


def run(state: AnalysisState) -> AnalysisState:
    if not state.news_items:
        logger.warning("Step 6 | no news items — skipping sentiment analysis.")
        state.overall_sentiment = 0.0
        return state

    client   = get_watsonx_client()
    template = load_prompt("sentiment_classification")
    logger.info("Step 6 | classifying %d news items", len(state.news_items))

    classified: list[NewsItem] = []
    for item in state.news_items:
        try:
            classified.append(_classify_item(client, item, template))
        except Exception as exc:
            logger.warning("  sentiment classification failed for '%s': %s", item.title[:60], exc)
            item.sentiment  = "neutral"
            item.confidence = 0.5
            classified.append(item)

    state.news_items = classified

    # ── Build sentiment index (daily aggregated score) ─────────────────────
    daily_scores: dict[str, list[float]] = defaultdict(list)
    for item in classified:
        score = _SENTIMENT_MAP.get(item.sentiment, 0.0) * item.confidence
        date_key = item.published[:10] if item.published else datetime.today().strftime("%Y-%m-%d")
        daily_scores[date_key].append(score)

    sentiment_index = [
        {"date": date, "score": round(sum(scores) / len(scores), 3)}
        for date, scores in sorted(daily_scores.items())
    ]
    state.sentiment_index = sentiment_index

    # ── Overall sentiment: weighted mean across all items ──────────────────
    all_scores = [_SENTIMENT_MAP.get(i.sentiment, 0.0) * i.confidence for i in classified]
    state.overall_sentiment = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

    pos = sum(1 for i in classified if i.sentiment == "positive")
    neg = sum(1 for i in classified if i.sentiment == "negative")
    logger.info(
        "Step 6 done | positive=%d negative=%d overall=%.3f",
        pos, neg, state.overall_sentiment,
    )
    return state
