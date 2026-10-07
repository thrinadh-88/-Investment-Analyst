"""
Step 5 — Competitor Benchmarking
Builds a ranked comparison table of all tickers (primary + peers),
computes sector medians, and assigns relative-strength z-scores.
"""
from __future__ import annotations

import logging

import numpy as np

from app.models.state import AnalysisState, PeerRow

logger = logging.getLogger(__name__)

_RATIO_FIELDS = ["pe", "pb", "roe", "debt_to_equity", "net_margin"]


def _safe(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def run(state: AnalysisState) -> AnalysisState:
    all_tickers = list(dict.fromkeys(state.primary_tickers + state.peer_tickers))
    logger.info("Step 5 | benchmarking %d tickers", len(all_tickers))

    rows: list[PeerRow] = []
    for ticker in all_tickers:
        r = state.ratios.get(ticker)
        k = state.kpis.get(ticker)
        rows.append(PeerRow(
            ticker         = ticker,
            pe             = _safe(r.pe)             if r else None,
            pb             = _safe(r.pb)             if r else None,
            roe            = _safe(r.roe)            if r else None,
            debt_to_equity = _safe(r.debt_to_equity) if r else None,
            net_margin     = _safe(k.net_margin)     if k else None,
        ))

    # ── Sector medians ────────────────────────────────────────────────────────
    medians: dict[str, float | None] = {}
    for field in _RATIO_FIELDS:
        vals = [getattr(row, field) for row in rows if getattr(row, field) is not None]
        medians[field] = float(np.median(vals)) if vals else None
    state.sector_median = medians

    # ── Relative-strength z-score ─────────────────────────────────────────────
    # Higher ROE, lower PE, lower D/E = better → composite signal
    # Normalise each metric and combine into a single score
    def _normalise(values: list[float | None], higher_is_better: bool = True) -> list[float]:
        clean = [v if v is not None else float("nan") for v in values]
        arr   = np.array(clean, dtype=float)
        mean  = np.nanmean(arr)
        std   = np.nanstd(arr) + 1e-9
        z     = (arr - mean) / std
        return list(-z if not higher_is_better else z)  # invert for "lower is better"

    n = len(rows)
    pe_z  = _normalise([r.pe             for r in rows], higher_is_better=False)
    roe_z = _normalise([r.roe            for r in rows], higher_is_better=True)
    de_z  = _normalise([r.debt_to_equity for r in rows], higher_is_better=False)
    nm_z  = _normalise([r.net_margin     for r in rows], higher_is_better=True)

    for i, row in enumerate(rows):
        components = [pe_z[i], roe_z[i], de_z[i], nm_z[i]]
        valid = [c for c in components if not np.isnan(c)]
        row.rel_strength = round(float(np.mean(valid)), 3) if valid else 0.0

    # Sort by relative strength descending
    rows.sort(key=lambda r: r.rel_strength or 0.0, reverse=True)

    state.benchmark_table = rows
    logger.info(
        "Step 5 done | %d rows | medians: PE=%.1f ROE=%.3f",
        len(rows),
        medians.get("pe") or 0,
        medians.get("roe") or 0,
    )
    return state
