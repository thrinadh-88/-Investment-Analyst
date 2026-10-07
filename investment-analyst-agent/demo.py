"""
demo.py — End-to-end demo of the Investment Analyst Agent.
Analyses TCS.NS + INFY.NS (primary) vs WIPRO.NS + HCLTECH.NS (peers)
in MOCK_MODE so it runs without any API credentials.

Usage:
    cd investment-analyst-agent
    python demo.py
"""
from __future__ import annotations

import sys
import os
import json
import logging
from pathlib import Path

# Ensure stdout can handle Unicode on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force mock mode
os.environ["MOCK_MODE"] = "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("demo")


def main() -> None:
    from app.models.state import AnalysisState
    from app.agents.orchestrator import Orchestrator

    print("\n" + "=" * 60)
    print("  Investment Analyst Agent — Demo Run")
    print("  Primary:  TCS.NS, INFY.NS")
    print("  Peers:    WIPRO.NS, HCLTECH.NS")
    print("  Mode:     MOCK (no credentials needed)")
    print("=" * 60 + "\n")

    state = AnalysisState(
        primary_tickers=["TCS.NS", "INFY.NS"],
        peer_tickers=["WIPRO.NS", "HCLTECH.NS"],
        period="1y",
        mock_mode=True,
    )

    def progress(step_num: int, status: str, message: str) -> None:
        icon = {"running": "▶", "done": "✓", "failed": "✗", "skipped": "–"}.get(status, "·")
        print(f"  [{icon}] Step {step_num:02d}  {status.upper():8s}  {message}")

    orch   = Orchestrator()
    result = orch.run(state, on_progress=progress)

    print("\n" + "─" * 60)
    print("  RESULTS")
    print("─" * 60)

    for ticker in result.primary_tickers:
        rec = result.recommendations.get(ticker)
        if rec:
            print(f"\n  {ticker}")
            print(f"  Rating:     {rec.rating}  (confidence {rec.confidence:.0%})")
            print(f"  Summary:    {rec.summary}")
            print(f"  Reasons:")
            for r in rec.reasons:
                print(f"    • {r}")
            print(f"  Risks:")
            for r in rec.risks:
                print(f"    ⚠ {r}")

    print(f"\n  Overall Sentiment:   {result.overall_sentiment:+.3f}")
    print(f"  Data Quality Score:  {result.data_quality_score:.0f}/100")
    print(f"  Report saved to:     {result.report_url}")

    print("\n  Risk Indicators:")
    for risk in result.risk_indicators:
        print(f"    {risk.level:6s} — {risk.name}: {risk.explanation}")

    print("\n  Peer Benchmarking (relative strength ranking):")
    for row in result.benchmark_table:
        print(f"    {row.ticker:12s}  score={row.rel_strength:+.3f}  PE={row.pe}  ROE={row.roe}")

    print("\n" + "=" * 60)
    print("  Demo complete.  To run with real credentials:")
    print("  1. Copy .env.example → .env and fill in your keys")
    print("  2. Set MOCK_MODE=false")
    print("  3. streamlit run dashboard/app.py")
    print("=" * 60 + "\n")

    # Validate all steps completed
    failed = [sr for sr in result.step_results if sr.status.value in ("failed", "skipped")]
    if failed:
        print(f"  ⚠  {len(failed)} step(s) failed/skipped — check logs above.")
        sys.exit(1)
    else:
        print("  ✓  All 9 pipeline steps completed successfully.")


if __name__ == "__main__":
    main()
