"""
Step 3 — KPI Extraction
Extracts structured KPIs from fundamentals/financials where available.
Falls back to Granite LLM for any ticker whose structured data is insufficient.
"""
from __future__ import annotations

import logging
from typing import Any

from app.models.state import AnalysisState, KPIs
from app.services.watsonx_client import get_watsonx_client

logger = logging.getLogger(__name__)

_MIN_KPI_FIELDS = 4   # require at least this many non-None KPI fields before skipping Granite


def _extract_from_structured(fund: dict[str, Any]) -> KPIs:
    """Pull KPIs directly from yfinance fundamentals dict."""
    return KPIs(
        revenue          = fund.get("totalRevenue"),
        net_income       = fund.get("netIncomeToCommon"),
        ebitda           = fund.get("ebitda"),
        operating_margin = fund.get("operatingMargins"),
        net_margin       = fund.get("profitMargins"),
        eps              = fund.get("trailingEps"),
        total_debt       = fund.get("totalDebt"),
        free_cash_flow   = fund.get("freeCashflow"),
        revenue_yoy      = fund.get("revenueGrowth"),
        net_income_yoy   = fund.get("earningsGrowth"),
    )


def _count_non_none(kpis: KPIs) -> int:
    return sum(1 for v in kpis.model_dump().values() if v is not None)


def _build_kpi_prompt(ticker: str, news_text: str, fund_text: str) -> str:
    from app.prompts import load_prompt
    template = load_prompt("kpi_extraction")
    return template.format(ticker=ticker, news_text=news_text, fund_text=fund_text)


def run(state: AnalysisState) -> AnalysisState:
    client = get_watsonx_client()
    all_tickers = list(dict.fromkeys(state.primary_tickers + state.peer_tickers))
    logger.info("Step 3 | extracting KPIs for %d tickers", len(all_tickers))

    for ticker in all_tickers:
        fund = state.fundamentals.get(ticker, {})
        kpis = _extract_from_structured(fund)
        non_none = _count_non_none(kpis)

        if non_none < _MIN_KPI_FIELDS:
            logger.info("  %s — only %d KPI fields from structured data; trying Granite.", ticker, non_none)
            # Build context text from news headlines
            news_text = " | ".join(
                item.title for item in state.news_items
                if ticker.replace(".NS", "").replace(".BO", "").lower() in item.title.lower()
            )[:1500]
            fund_text = str({k: v for k, v in fund.items() if v is not None})[:1000]

            try:
                from app.prompts import load_prompt
                template = load_prompt("kpi_extraction")
                prompt   = template.format(
                    ticker    = ticker,
                    news_text = news_text or "No news available.",
                    fund_text = fund_text or "No structured data available.",
                )
                result = client.extract_json(prompt)
                if result:
                    # Merge Granite results into structured KPIs (fill None gaps only)
                    kpis_dict = kpis.model_dump()
                    for field, value in result.items():
                        if field in kpis_dict and kpis_dict[field] is None and value is not None:
                            try:
                                kpis_dict[field] = float(value)
                            except (TypeError, ValueError):
                                pass
                    kpis = KPIs(**kpis_dict)
                    logger.debug("  %s — Granite filled %d additional KPI fields.", ticker, _count_non_none(kpis) - non_none)
            except Exception as exc:
                logger.warning("  %s — Granite KPI extraction failed: %s", ticker, exc)

        state.kpis[ticker] = kpis
        logger.debug("  %s KPIs: %d non-None fields", ticker, _count_non_none(kpis))

    logger.info("Step 3 done | tickers with KPIs: %d", len(state.kpis))
    return state
