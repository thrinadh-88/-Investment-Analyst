"""
tests/test_orchestrator.py — unit tests for the Orchestrator.
Uses pytest-mock to stub out individual step functions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.models.state import AnalysisState, StepStatus
from app.agents.orchestrator import Orchestrator, PIPELINE_STEPS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_state() -> AnalysisState:
    return AnalysisState(
        primary_tickers=["AAPL"],
        peer_tickers=["MSFT"],
        mock_mode=True,
    )


def _noop(state: AnalysisState) -> AnalysisState:
    """A no-op step that just returns state unchanged."""
    return state


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_all_steps_run_successfully(mocker):
    """All 9 step functions should be called and state.step_results should have 9 DONE entries."""
    for num, _name, _fn in PIPELINE_STEPS:
        mocker.patch(
            f"app.agents.step{num}_{'_'.join(_name.lower().split())}.run"
            if False else f"app.agents.orchestrator.PIPELINE_STEPS",
            new=None,
        )

    # Patch via the orchestrator module's PIPELINE_STEPS list directly
    patched_steps = [(num, name, lambda s, _n=num: s) for num, name, _ in PIPELINE_STEPS]
    mocker.patch("app.agents.orchestrator.PIPELINE_STEPS", patched_steps)

    state  = _minimal_state()
    result = Orchestrator().run(state)

    assert len(result.step_results) == 9
    for sr in result.step_results:
        assert sr.status == StepStatus.DONE, f"Step {sr.step_number} [{sr.step_name}] was not DONE: {sr.status}"


def test_step_failure_marks_failed_and_continues(mocker):
    """A failure in Step 3 should mark it FAILED but allow Steps 4–9 to continue."""
    def _fail_step3(state, _num=3):
        raise RuntimeError("Simulated KPI failure")

    patched_steps = [
        (num, name, (_fail_step3 if num == 3 else (lambda s, _n=num: s)))
        for num, name, _ in PIPELINE_STEPS
    ]
    mocker.patch("app.agents.orchestrator.PIPELINE_STEPS", patched_steps)

    state  = _minimal_state()
    result = Orchestrator().run(state)

    step3 = next(sr for sr in result.step_results if sr.step_number == 3)
    assert step3.status == StepStatus.FAILED

    # All other steps should be DONE
    for sr in result.step_results:
        if sr.step_number != 3:
            assert sr.status == StepStatus.DONE, f"Step {sr.step_number} should be DONE, got {sr.status}"


def test_step1_failure_skips_remaining(mocker):
    """A hard failure in Step 1 should skip all subsequent steps."""
    def _fail_step1(state):
        raise RuntimeError("Data gathering failed")

    patched_steps = [
        (num, name, (_fail_step1 if num == 1 else (lambda s, _n=num: s)))
        for num, name, _ in PIPELINE_STEPS
    ]
    mocker.patch("app.agents.orchestrator.PIPELINE_STEPS", patched_steps)

    state  = _minimal_state()
    result = Orchestrator().run(state)

    step1 = next(sr for sr in result.step_results if sr.step_number == 1)
    assert step1.status == StepStatus.FAILED

    for sr in result.step_results:
        if sr.step_number > 1:
            assert sr.status == StepStatus.SKIPPED, f"Step {sr.step_number} should be SKIPPED, got {sr.status}"


def test_step_results_populated():
    """After a run, all step_results have step_number and step_name set."""
    # Run with real mock-mode data (no mocker needed)
    state  = _minimal_state()
    result = Orchestrator().run(state)

    assert len(result.step_results) == 9
    for i, sr in enumerate(result.step_results, 1):
        assert sr.step_number == i
        assert sr.step_name != ""


def test_on_progress_callback_called(mocker):
    """on_progress callback should be called twice per step (RUNNING + DONE/FAILED)."""
    patched_steps = [(num, name, lambda s, _n=num: s) for num, name, _ in PIPELINE_STEPS]
    mocker.patch("app.agents.orchestrator.PIPELINE_STEPS", patched_steps)

    calls: list[tuple] = []
    def callback(step_num: int, status: str, message: str) -> None:
        calls.append((step_num, status))

    state = _minimal_state()
    Orchestrator().run(state, on_progress=callback)

    # Should have at least 9 DONE callbacks
    done_calls = [(n, s) for n, s in calls if s == "done"]
    assert len(done_calls) == 9


def test_run_timestamps_set():
    """run_started_at and run_finished_at should be populated after run."""
    state  = _minimal_state()
    result = Orchestrator().run(state)
    assert result.run_started_at  is not None
    assert result.run_finished_at is not None
    assert result.run_finished_at >= result.run_started_at
