"""
agents/orchestrator.py — Sequential pipeline runner.

Runs all 9 steps in order, passing and mutating a shared AnalysisState.
Supports:
  - Per-step status tracking (StepResult list on state)
  - Retries with exponential back-off (up to 2 retries)
  - Graceful degradation (step failure → FAILED status, pipeline continues)
  - Fatal early exit if Step 1 fails completely
  - on_progress callback for live UI updates: fn(step_num, status, message)
  - Structured logging for every step
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from app.agents import (
    step1_data_gathering,
    step2_validation,
    step3_kpi_extraction,
    step4_ratios,
    step5_benchmarking,
    step6_sentiment,
    step7_predictive,
    step8_recommendations,
    step9_report,
)
from app.models.state import AnalysisState, StepResult, StepStatus

logger = logging.getLogger(__name__)

# ── Pipeline definition ───────────────────────────────────────────────────────

PIPELINE_STEPS: list[tuple[int, str, Any]] = [
    (1, "Data Gathering",         step1_data_gathering.run),
    (2, "Validation",             step2_validation.run),
    (3, "KPI Extraction",         step3_kpi_extraction.run),
    (4, "Ratio Calculation",      step4_ratios.run),
    (5, "Competitor Benchmarking",step5_benchmarking.run),
    (6, "Sentiment Analysis",     step6_sentiment.run),
    (7, "Predictive Insights",    step7_predictive.run),
    (8, "Recommendations",        step8_recommendations.run),
    (9, "Report Delivery",        step9_report.run),
]

_MAX_RETRIES   = 2
_RETRY_DELAYS  = [0.5, 1.0]   # seconds between retries


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Runs the investment analysis pipeline sequentially.

    Usage:
        state = AnalysisState(primary_tickers=["TCS.NS"], peer_tickers=["INFY.NS"])
        result = Orchestrator().run(state, on_progress=my_callback)
    """

    def run(
        self,
        state: AnalysisState,
        on_progress: Callable[[int, str, str], None] | None = None,
    ) -> AnalysisState:
        """
        Execute all pipeline steps and return the populated state.

        Args:
            state:       Pre-populated AnalysisState with tickers + params.
            on_progress: Optional callback(step_number, status_str, message).
        """
        state.run_started_at = datetime.utcnow()
        state.step_results   = [
            StepResult(step_number=num, step_name=name)
            for num, name, _ in PIPELINE_STEPS
        ]

        logger.info(
            "Orchestrator starting | tickers=%s peers=%s period=%s mock=%s",
            state.primary_tickers,
            state.peer_tickers,
            state.period,
            state.mock_mode,
        )

        for idx, (step_num, step_name, step_fn) in enumerate(PIPELINE_STEPS):
            result = state.step_results[idx]
            self._notify(on_progress, step_num, StepStatus.RUNNING, f"Running {step_name}…")

            success = False
            last_exc: Exception | None = None

            for attempt in range(1 + _MAX_RETRIES):
                if attempt > 0:
                    delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
                    logger.warning(
                        "  Step %d retry %d/%d after %.1fs", step_num, attempt, _MAX_RETRIES, delay
                    )
                    time.sleep(delay)

                t0 = time.perf_counter()
                result.started_at = datetime.utcnow()

                try:
                    state = step_fn(state)
                    result.duration_ms  = (time.perf_counter() - t0) * 1000
                    result.status       = StepStatus.DONE
                    result.finished_at  = datetime.utcnow()
                    result.message      = f"Completed in {result.duration_ms:.0f} ms"
                    success = True
                    logger.info(
                        "  ✓ Step %d [%s] %.0f ms",
                        step_num, step_name, result.duration_ms
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    result.duration_ms = (time.perf_counter() - t0) * 1000
                    logger.warning(
                        "  ✗ Step %d [%s] attempt %d failed: %s",
                        step_num, step_name, attempt + 1, exc,
                    )

            if not success:
                result.status      = StepStatus.FAILED
                result.message     = str(last_exc)
                result.finished_at = datetime.utcnow()
                state.error_summary.append(f"Step {step_num} [{step_name}]: {last_exc}")
                self._notify(on_progress, step_num, StepStatus.FAILED, str(last_exc))
                logger.error(
                    "  ✗ Step %d [%s] permanently failed: %s", step_num, step_name, last_exc
                )

                # Step 1 failure is fatal — skip remaining steps
                if step_num == 1:
                    logger.error("Step 1 (Data Gathering) failed — aborting pipeline.")
                    for remaining in state.step_results[idx + 1:]:
                        remaining.status  = StepStatus.SKIPPED
                        remaining.message = "Skipped: Step 1 failed."
                    self._notify(on_progress, step_num, StepStatus.FAILED, "Pipeline aborted: data gathering failed.")
                    break

                # Other steps: continue pipeline
                continue

            self._notify(on_progress, step_num, StepStatus.DONE, result.message)

        state.run_finished_at = datetime.utcnow()
        duration_s = (state.run_finished_at - state.run_started_at).total_seconds()
        logger.info(
            "Orchestrator finished | %.1fs | errors=%d",
            duration_s,
            len(state.error_summary),
        )
        return state

    @staticmethod
    def _notify(
        callback: Callable[[int, str, str], None] | None,
        step_num: int,
        status: StepStatus,
        message: str,
    ) -> None:
        if callback:
            try:
                callback(step_num, status.value, message)
            except Exception as exc:
                logger.debug("on_progress callback error: %s", exc)
