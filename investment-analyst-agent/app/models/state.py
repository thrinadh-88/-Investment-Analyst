"""
app/models/state.py — shared Pydantic state object passed through every
pipeline step.  Each step reads from and writes to this object only; no
step has side-effects on global state.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Step lifecycle ────────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    SKIPPED = "skipped"


class StepResult(BaseModel):
    step_number: int
    step_name:   str
    status:      StepStatus = StepStatus.PENDING
    message:     str        = ""
    duration_ms: float      = 0.0
    started_at:  datetime | None = None
    finished_at: datetime | None = None


# ── Per-step output schemas ───────────────────────────────────────────────────

class NewsItem(BaseModel):
    title:      str
    summary:    str = ""
    published:  str = ""
    source:     str = ""
    url:        str = ""
    sentiment:  str = "neutral"   # populated in Step 6
    confidence: float = 0.0
    rationale:  str = ""


class KPIs(BaseModel):
    revenue:           float | None = None
    net_income:        float | None = None
    ebitda:            float | None = None
    operating_margin:  float | None = None
    net_margin:        float | None = None
    eps:               float | None = None
    total_debt:        float | None = None
    free_cash_flow:    float | None = None
    revenue_yoy:       float | None = None   # % growth
    net_income_yoy:    float | None = None


class Ratios(BaseModel):
    pe:               float | None = None
    pb:               float | None = None
    roe:              float | None = None
    roa:              float | None = None
    debt_to_equity:   float | None = None
    current_ratio:    float | None = None
    interest_coverage: float | None = None
    beta:             float | None = None
    volatility_ann:   float | None = None   # annualised std dev
    max_drawdown:     float | None = None


class RiskIndicator(BaseModel):
    name:        str
    level:       str   # Low / Medium / High
    score:       float # 0–1
    explanation: str


class Recommendation(BaseModel):
    rating:      str   # Buy / Hold / Reduce
    confidence:  float # 0–1
    summary:     str
    reasons:     list[str] = Field(default_factory=list)
    risks:       list[str] = Field(default_factory=list)
    citations:   list[str] = Field(default_factory=list)


class PeerRow(BaseModel):
    ticker:         str
    pe:             float | None = None
    pb:             float | None = None
    roe:            float | None = None
    debt_to_equity: float | None = None
    net_margin:     float | None = None
    rel_strength:   float | None = None   # z-score vs peers


# ── Master state ──────────────────────────────────────────────────────────────

class AnalysisState(BaseModel):
    # ── Inputs ────────────────────────────────────────────────────────────────
    primary_tickers: list[str]  = Field(default_factory=list)
    peer_tickers:    list[str]  = Field(default_factory=list)
    period:          str        = "1y"    # yfinance period string
    mock_mode:       bool       = False

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    price_history:      dict[str, Any] = Field(default_factory=dict)  # ticker -> records
    fundamentals:       dict[str, Any] = Field(default_factory=dict)  # ticker -> info dict
    financials:         dict[str, Any] = Field(default_factory=dict)  # ticker -> income stmt
    news_items:         list[NewsItem] = Field(default_factory=list)
    analyst_opinions:   dict[str, Any] = Field(default_factory=dict)

    # ── Step 2: Validation ────────────────────────────────────────────────────
    data_quality_score: float      = 100.0
    validation_flags:   list[str]  = Field(default_factory=list)

    # ── Step 3: KPIs ──────────────────────────────────────────────────────────
    kpis: dict[str, KPIs] = Field(default_factory=dict)  # ticker -> KPIs

    # ── Step 4: Ratios ────────────────────────────────────────────────────────
    ratios: dict[str, Ratios] = Field(default_factory=dict)  # ticker -> Ratios

    # ── Step 5: Benchmarking ──────────────────────────────────────────────────
    benchmark_table:  list[PeerRow] = Field(default_factory=list)
    sector_median:    dict[str, float | None] = Field(default_factory=dict)

    # ── Step 6: Sentiment ─────────────────────────────────────────────────────
    sentiment_index:  list[dict[str, Any]] = Field(default_factory=list)  # [{date, score}]
    overall_sentiment: float = 0.0  # -1 to +1

    # ── Step 7: Predictive ────────────────────────────────────────────────────
    trend_signals:    dict[str, Any] = Field(default_factory=dict)
    price_forecast:   dict[str, Any] = Field(default_factory=dict)
    risk_indicators:  list[RiskIndicator] = Field(default_factory=list)

    # ── Step 8: Recommendation ────────────────────────────────────────────────
    recommendations: dict[str, Recommendation] = Field(default_factory=dict)  # ticker -> rec

    # ── Step 9: Report ────────────────────────────────────────────────────────
    report_url:   str = ""
    report_json:  str = ""
    report_html:  str = ""

    # ── Orchestrator metadata ─────────────────────────────────────────────────
    step_results: list[StepResult] = Field(default_factory=list)
    run_started_at:  datetime | None = None
    run_finished_at: datetime | None = None
    error_summary:   list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
