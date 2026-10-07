"""
Step 8 — Recommendation Generation
Uses IBM Granite to produce a Buy/Hold/Reduce recommendation grounded
in computed state data. Parses structured JSON output.
"""
from __future__ import annotations

import logging

from app.models.state import AnalysisState, Recommendation
from app.prompts import load_prompt
from app.services.watsonx_client import get_watsonx_client

logger = logging.getLogger(__name__)


def _build_prompt(state: AnalysisState, ticker: str) -> str:
    template = load_prompt("recommendation")

    # KPIs text
    kpis = state.kpis.get(ticker)
    kpis_text = "\n".join(
        f"  {k}: {v}" for k, v in (kpis.model_dump().items() if kpis else {}.items()) if v is not None
    ) or "  No KPI data available."

    # Ratios text
    ratios = state.ratios.get(ticker)
    ratios_text = "\n".join(
        f"  {k}: {v}" for k, v in (ratios.model_dump().items() if ratios else {}.items()) if v is not None
    ) or "  No ratio data available."

    # Benchmark text
    bench_rows = [r for r in state.benchmark_table if r.ticker == ticker]
    if bench_rows:
        br = bench_rows[0]
        benchmark_text = (
            f"  Relative strength score vs peers: {br.rel_strength}\n"
            f"  Sector median P/E: {state.sector_median.get('pe')}\n"
            f"  Sector median ROE: {state.sector_median.get('roe')}"
        )
    else:
        benchmark_text = "  No benchmark data available."

    # Risk text
    risk_text = "\n".join(
        f"  {r.name}: {r.level} (score={r.score:.2f}) — {r.explanation}"
        for r in state.risk_indicators
    ) or "  No risk data available."

    # Analyst text
    ao = state.analyst_opinions.get(ticker, {})
    analyst_text = (
        f"Recommendation key: {ao.get('recommendationKey', 'N/A')}, "
        f"Target price: {ao.get('targetMeanPrice', 'N/A')}"
    )

    return template.format(
        ticker          = ticker,
        period          = state.period,
        kpis_text       = kpis_text,
        ratios_text     = ratios_text,
        benchmark_text  = benchmark_text,
        sentiment_score = f"{state.overall_sentiment:.3f}",
        risk_text       = risk_text,
        analyst_text    = analyst_text,
    )


def _parse_recommendation(data: dict) -> Recommendation:
    rating = str(data.get("rating", "Hold")).strip()
    if rating not in ("Buy", "Hold", "Reduce"):
        rating = "Hold"
    return Recommendation(
        rating     = rating,
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
        summary    = str(data.get("summary", "")),
        reasons    = [str(r) for r in data.get("reasons", [])],
        risks      = [str(r) for r in data.get("risks",   [])],
        citations  = [str(c) for c in data.get("citations", [])],
    )


def run(state: AnalysisState) -> AnalysisState:
    client = get_watsonx_client()
    logger.info("Step 8 | generating recommendations for %d tickers", len(state.primary_tickers))

    for ticker in state.primary_tickers:
        try:
            prompt = _build_prompt(state, ticker)
            result = client.extract_json(prompt, max_tokens=800)
            if result:
                state.recommendations[ticker] = _parse_recommendation(result)
                logger.info(
                    "  %s → %s (confidence=%.2f)",
                    ticker,
                    state.recommendations[ticker].rating,
                    state.recommendations[ticker].confidence,
                )
            else:
                logger.warning("  %s — Granite returned no parseable JSON.", ticker)
                state.recommendations[ticker] = Recommendation(
                    rating="Hold",
                    confidence=0.4,
                    summary="Insufficient data for a confident recommendation.",
                    reasons=["Data quality issues prevented full analysis"],
                    risks=["Data gaps may hide underlying risks"],
                    citations=[],
                )
        except Exception as exc:
            logger.error("  %s — recommendation generation failed: %s", ticker, exc)
            state.recommendations[ticker] = Recommendation(
                rating="Hold",
                confidence=0.3,
                summary="Analysis failed due to a technical error.",
                reasons=[],
                risks=[str(exc)],
                citations=[],
            )

    logger.info("Step 8 done | %d recommendations generated", len(state.recommendations))
    return state
