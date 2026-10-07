"""
dashboard/app.py — Streamlit Investment Analyst Dashboard.
Entry point: streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from app.config import settings
from app.models.state import AnalysisState, StepStatus
from app.agents.orchestrator import Orchestrator, PIPELINE_STEPS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Analyst Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.metric-card {
    background: #f7f8fa; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 16px 20px; text-align: center;
}
.rating-Buy    {color: #057a55; font-size: 1.6rem; font-weight: 800;}
.rating-Hold   {color: #c27803; font-size: 1.6rem; font-weight: 800;}
.rating-Reduce {color: #c81e1e; font-size: 1.6rem; font-weight: 800;}
.disclaimer-box {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 6px; padding: 10px 16px;
    font-size: 0.82rem; color: #856404; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for _key, _default in [
    ("analysis_state", None),
    ("step_log", []),
    ("running", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Cached chart helpers ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_price_chart(records_key: str, records: list, ticker: str, forecast_key: str, forecast: dict) -> go.Figure:
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                             name="Close", line=dict(color="#1a56db", width=1.5)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"],
                             name="MA20", line=dict(color="#f59e0b", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"],
                             name="MA50", line=dict(color="#ef4444", width=1, dash="dot")))
    if forecast.get("forecast_30d"):
        last_date   = df["Date"].iloc[-1]
        future_date = last_date + pd.Timedelta(days=30)
        fig.add_trace(go.Scatter(
            x=[last_date, future_date],
            y=[df["Close"].iloc[-1], forecast["forecast_30d"]],
            name="30d Forecast",
            line=dict(color="#7c3aed", width=2, dash="dash"),
        ))
    fig.update_layout(
        title=f"{ticker} — Price & Trend",
        xaxis_title="Date", yaxis_title="Price",
        legend=dict(orientation="h", y=1.1),
        height=360, margin=dict(t=50, b=20),
    )
    return fig


@st.cache_data(show_spinner=False)
def _build_bench_chart(cache_key: str, bench_records: list) -> go.Figure:
    bench_df = pd.DataFrame(bench_records)
    fig = go.Figure()
    for metric, label in [("pe", "P/E"), ("roe", "ROE"), ("net_margin", "Net Margin")]:
        if metric in bench_df.columns:
            vals = bench_df[metric].fillna(0)
            if metric in ("roe", "net_margin"):
                vals = vals * 100
            fig.add_trace(go.Bar(name=label, x=bench_df["ticker"], y=vals))
    fig.update_layout(barmode="group", title="Metric Comparison vs Peers",
                      height=320, margin=dict(t=50, b=20))
    return fig


@st.cache_data(show_spinner=False)
def _build_sentiment_chart(cache_key: str, sentiment_index: list) -> go.Figure:
    sent_df = pd.DataFrame(sentiment_index)
    sent_df["date"] = pd.to_datetime(sent_df["date"])
    fig = px.line(sent_df, x="date", y="score",
                  title="Daily Sentiment Score (-1 Negative → +1 Positive)",
                  labels={"score": "Sentiment", "date": "Date"},
                  color_discrete_sequence=["#1a56db"])
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(height=260, margin=dict(t=50, b=20))
    return fig


@st.cache_data(show_spinner=False)
def _build_risk_chart(cache_key: str, risk_records: list) -> go.Figure:
    risk_df = pd.DataFrame(risk_records)
    fig = px.bar(risk_df, x="Risk", y="Score", color="Score",
                 color_continuous_scale=[(0, "#057a55"), (0.5, "#f59e0b"), (1, "#c81e1e")],
                 range_color=[0, 1], title="Risk Scores")
    fig.update_layout(height=260, coloraxis_showscale=False, margin=dict(t=50, b=20))
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────
def _parse_tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Investment Analyst")
    st.caption("IBM Granite · IBM Cloud")
    st.divider()

    primary_input = st.text_input("Primary Ticker(s)", value="TCS.NS, INFY.NS",
                                  help="Comma-separated, e.g. TCS.NS, INFY.NS or AAPL, MSFT")
    peer_input    = st.text_input("Peer Tickers", value="WIPRO.NS, HCLTECH.NS",
                                  help="Competitor tickers for benchmarking")
    period        = st.select_slider("Period", options=["1mo","3mo","6mo","1y","2y"], value="1y")
    mock_override = st.checkbox("Mock Mode (offline)", value=settings.mock_mode,
                                help="Synthetic data — no API keys needed")

    st.divider()
    run_button = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    st.divider()
    st.caption(f"Model: `{settings.granite_model_id}`")
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.analysis_state = None
        st.session_state.step_log       = []
        st.cache_data.clear()
        st.rerun()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_button:
    primary_tickers = _parse_tickers(primary_input)
    peer_tickers    = _parse_tickers(peer_input)

    if not primary_tickers:
        st.sidebar.error("Enter at least one primary ticker.")
        st.stop()

    os.environ["MOCK_MODE"] = "true" if mock_override else "false"
    st.cache_data.clear()   # bust cache on new run

    state = AnalysisState(
        primary_tickers=primary_tickers,
        peer_tickers=peer_tickers,
        period=period,
        mock_mode=mock_override,
    )
    st.session_state.step_log = [(num, "pending", name) for num, name, _ in PIPELINE_STEPS]

    # ── Live progress via st.status ───────────────────────────────────────────
    status_icons = {"pending":"⏳","running":"🔵","done":"✅","failed":"❌","skipped":"⏭️"}

    with st.status("Running analysis pipeline…", expanded=True) as status_box:
        step_placeholders = {}
        for num, name, _ in PIPELINE_STEPS:
            step_placeholders[num] = st.empty()
            step_placeholders[num].markdown(f"⏳ **Step {num}** — {name}")

        def on_progress(step_num: int, step_status: str, message: str) -> None:
            log = st.session_state.step_log
            for i, (n, _, name) in enumerate(log):
                if n == step_num:
                    log[i] = (n, step_status, name)
                    break
            icon = status_icons.get(step_status, "⏳")
            ph   = step_placeholders.get(step_num)
            if ph:
                label = dict(PIPELINE_STEPS)[step_num] if False else \
                        next((nm for s, nm, _ in PIPELINE_STEPS if s == step_num), "")
                if step_status == "running":
                    ph.markdown(f"🔵 **Step {step_num}** — {label}  `running…`")
                elif step_status == "done":
                    ph.markdown(f"✅ **Step {step_num}** — {label}  `{message}`")
                elif step_status == "failed":
                    ph.markdown(f"❌ **Step {step_num}** — {label}  `{message}`")
                else:
                    ph.markdown(f"{icon} **Step {step_num}** — {label}")

        final = Orchestrator().run(state, on_progress=on_progress)
        errors = [sr for sr in final.step_results if sr.status == StepStatus.FAILED]
        if errors:
            status_box.update(label=f"Analysis complete — {len(errors)} step(s) had issues", state="error")
        else:
            status_box.update(label="Analysis complete ✅", state="complete")

    st.session_state.analysis_state = final
    st.session_state.running        = False

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 Investment Analyst Agent")
st.markdown("""
<div class="disclaimer-box">
⚠️ <strong>Disclaimer:</strong> AI-generated for <strong>educational purposes only</strong> —
not financial advice. Always consult a qualified financial advisor.
</div>
""", unsafe_allow_html=True)

state: AnalysisState | None = st.session_state.analysis_state

if state is None:
    st.info("👈 Enter tickers in the sidebar and click **Run Analysis** to begin.")
    st.stop()

primary = state.primary_tickers
anchor  = primary[0] if primary else ""

# ── 1. Summary cards ──────────────────────────────────────────────────────────
st.subheader("💡 Investment Summary")
rec           = state.recommendations.get(anchor)
sentiment_val = state.overall_sentiment
sent_label    = "Positive" if sentiment_val > 0.1 else ("Negative" if sentiment_val < -0.1 else "Neutral")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Rating",       rec.rating             if rec else "—")
c2.metric("🔬 Confidence",   f"{rec.confidence:.0%}" if rec else "—")
c3.metric("📊 Data Quality", f"{state.data_quality_score:.0f}/100")
c4.metric("📰 Sentiment",    sent_label, delta=f"{sentiment_val:+.2f}")

if rec and rec.summary:
    rating_css = f"rating-{rec.rating}"
    st.markdown(
        f"<div class='metric-card'>"
        f"<span class='{rating_css}'>{rec.rating}</span><br>"
        f"<small>{rec.summary}</small></div>",
        unsafe_allow_html=True,
    )
st.divider()

# ── 2. Price & Trend ──────────────────────────────────────────────────────────
st.subheader("📈 Price & Trend")
for tab, ticker in zip(st.tabs(primary), primary):
    with tab:
        records  = state.price_history.get(ticker, [])
        forecast = state.price_forecast.get(ticker, {})
        if not records:
            st.warning(f"No price data for {ticker}.")
            continue
        records_key  = str(hash(str(records)))
        forecast_key = str(hash(str(forecast)))
        st.plotly_chart(
            _build_price_chart(records_key, records, ticker, forecast_key, forecast),
            use_container_width=True,
        )
        ts = state.trend_signals.get(ticker, {})
        if ts:
            ca, cb, cc = st.columns(3)
            ca.metric("MA20",   ts.get("ma20", "—"))
            cb.metric("MA50",   ts.get("ma50", "—"))
            cc.metric("Signal", ts.get("ma_signal", "—").title())
            if forecast.get("pct_change_30d") is not None:
                ca.metric("30d Forecast",    f"{forecast.get('forecast_30d','—')}")
                cb.metric("Projected Change", f"{forecast.get('pct_change_30d',0)*100:+.1f}%")
st.divider()

# ── 3. KPIs & Ratios ─────────────────────────────────────────────────────────
st.subheader("📋 Key Financial Metrics")
kpi_col, ratio_col = st.columns(2)

with kpi_col:
    st.markdown("**KPIs**")
    kpi_rows = []
    for t in primary:
        k = state.kpis.get(t)
        if k:
            kpi_rows.append({
                "Ticker":     t,
                "Revenue":    f"${k.revenue/1e9:.1f}B"      if k.revenue      else "—",
                "Net Income": f"${k.net_income/1e9:.1f}B"   if k.net_income   else "—",
                "Net Margin": f"{k.net_margin*100:.1f}%"    if k.net_margin   else "—",
                "EPS":        f"{k.eps:.2f}"                if k.eps          else "—",
                "Rev YoY":    f"{k.revenue_yoy*100:+.1f}%"  if k.revenue_yoy  else "—",
            })
    if kpi_rows:
        st.dataframe(pd.DataFrame(kpi_rows).set_index("Ticker"), use_container_width=True)

with ratio_col:
    st.markdown("**Ratios**")
    ratio_rows = [
        {
            "Ticker":     t,
            "P/E":        r.pe or "—",
            "P/B":        r.pb or "—",
            "ROE":        f"{r.roe*100:.1f}%"          if r.roe          else "—",
            "D/E":        r.debt_to_equity or "—",
            "Volatility": f"{r.volatility_ann*100:.1f}%" if r.volatility_ann else "—",
            "Max DD":     f"{r.max_drawdown*100:.1f}%"  if r.max_drawdown   else "—",
        }
        for t, r in state.ratios.items()
    ]
    if ratio_rows:
        st.dataframe(pd.DataFrame(ratio_rows).set_index("Ticker"), use_container_width=True)
st.divider()

# ── 4. Peer Benchmarking ─────────────────────────────────────────────────────
st.subheader("🏆 Peer Benchmarking")
if state.benchmark_table:
    bench_records = [r.model_dump() for r in state.benchmark_table]
    bench_key     = str(hash(str(bench_records)))
    st.plotly_chart(_build_bench_chart(bench_key, bench_records), use_container_width=True)
    bench_df = pd.DataFrame(bench_records)
    st.markdown("**Relative Strength Ranking**")
    rs_df = bench_df[["ticker","rel_strength"]].sort_values("rel_strength", ascending=False)
    st.dataframe(rs_df.rename(columns={"ticker":"Ticker","rel_strength":"Score"}),
                 use_container_width=True)
st.divider()

# ── 5. Sentiment ──────────────────────────────────────────────────────────────
st.subheader("📰 Market Sentiment")
if state.sentiment_index:
    sent_key = str(hash(str(state.sentiment_index)))
    st.plotly_chart(_build_sentiment_chart(sent_key, state.sentiment_index), use_container_width=True)
    with st.expander("📋 News & Sentiment Labels"):
        news_rows = [
            {"Date": n.published, "Title": n.title,
             "Sentiment": n.sentiment.title(),
             "Confidence": f"{n.confidence:.0%}",
             "Rationale": n.rationale}
            for n in state.news_items
        ]
        if news_rows:
            st.dataframe(pd.DataFrame(news_rows), use_container_width=True)
st.divider()

# ── 6. Risk Indicators ────────────────────────────────────────────────────────
st.subheader("⚠️ Risk Indicators")
if state.risk_indicators:
    colour_map = {"Low": "#057a55", "Medium": "#c27803", "High": "#c81e1e"}
    risk_cols  = st.columns(len(state.risk_indicators))
    for col, risk in zip(risk_cols, state.risk_indicators):
        colour = colour_map.get(risk.level, "#000")
        col.markdown(
            f"<div class='metric-card'>"
            f"<b style='font-size:12px'>{risk.name}</b><br>"
            f"<span style='color:{colour};font-size:1.3rem;font-weight:800'>{risk.level}</span><br>"
            f"<small style='color:#9ca3af'>{risk.score:.2f}</small><br>"
            f"<small>{risk.explanation}</small></div>",
            unsafe_allow_html=True,
        )
    risk_records = [{"Risk": r.name, "Score": r.score} for r in state.risk_indicators]
    risk_key     = str(hash(str(risk_records)))
    st.plotly_chart(_build_risk_chart(risk_key, risk_records), use_container_width=True)
st.divider()

# ── 7. Recommendations ────────────────────────────────────────────────────────
st.subheader("🤖 AI Recommendations")
st.markdown("""
<div class="disclaimer-box">
IBM Granite AI — grounded in computed data only. <strong>Educational purposes only.</strong>
</div>
""", unsafe_allow_html=True)

for ticker in primary:
    rec = state.recommendations.get(ticker)
    if not rec:
        st.warning(f"No recommendation for {ticker}.")
        continue
    colour = colour_map.get(rec.rating, "#000") if "colour_map" in dir() else \
             {"Buy":"#057a55","Hold":"#c27803","Reduce":"#c81e1e"}.get(rec.rating,"#000")
    with st.expander(f"**{ticker}** — {rec.rating} ({rec.confidence:.0%})", expanded=True):
        st.markdown(
            f"<span style='color:{colour};font-size:1.2rem;font-weight:800'>{rec.rating}</span>"
            f" — Confidence: {rec.confidence:.0%}",
            unsafe_allow_html=True,
        )
        if rec.summary:
            st.write(rec.summary)
        if rec.reasons:
            st.markdown("**Reasons:**")
            for r in rec.reasons:
                st.markdown(f"- {r}")
        if rec.risks:
            st.markdown("**Risks:**")
            for r in rec.risks:
                st.markdown(f"- ⚠️ {r}")
        if rec.citations:
            st.caption(f"Data used: {', '.join(rec.citations)}")
st.divider()

# ── 8. Download ───────────────────────────────────────────────────────────────
st.subheader("📥 Download Report")
dc1, dc2 = st.columns(2)
with dc1:
    if state.report_json:
        st.download_button("⬇️ JSON Report", data=state.report_json,
                           file_name=f"report_{'_'.join(primary)}.json",
                           mime="application/json")
with dc2:
    if state.report_html:
        st.download_button("⬇️ HTML Report", data=state.report_html,
                           file_name=f"report_{'_'.join(primary)}.html",
                           mime="text/html")
if state.report_url:
    st.caption(f"Saved: `{state.report_url}`")

if state.validation_flags:
    with st.expander("🔍 Validation Flags"):
        for flag in state.validation_flags:
            st.warning(flag)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-top:32px;font-size:11px;color:#9ca3af;
            border-top:1px solid #e5e7eb;padding-top:10px'>
  Investment Analyst Agent | IBM Granite + IBM Cloud | Made with IBM Bob
</div>
""", unsafe_allow_html=True)
