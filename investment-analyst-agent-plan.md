# Investment Analyst Agent — Plan

## Top-Level Overview

Build a **Sequential Task Agent – Investment Analyst** as a single-process
Streamlit application backed by a 9-step Python pipeline.  
Each pipeline step receives and mutates a shared, typed `AnalysisState` Pydantic
object managed by a central orchestrator.  
IBM Granite (via `ibm-watsonx-ai` SDK, region `au-syd`) powers three AI steps:
KPI extraction from unstructured text, news sentiment classification, and
final investment recommendation generation.  
IBM Cloud Object Storage stores the finished JSON/HTML reports.  
A `MOCK_MODE=true` env flag lets the full app run offline with synthetic data.

**Key constraint:** No FastAPI layer — Streamlit is the sole entry point and
calls the orchestrator directly inside its own process.

---

## Sub-Task 1 — Project Scaffold & Configuration

**Intent**  
Create the directory skeleton, dependency manifest, `.env.example`, `config.py`,
and the shared Pydantic state model. This is the foundation every other sub-task
builds on.

**Expected Outcomes**  
- `investment-analyst-agent/` tree matches the agreed structure  
- `requirements.txt` lists every required library at pinned-major versions  
- `config.py` loads all env vars via `python-dotenv` with typed defaults  
- `app/models/state.py` defines `AnalysisState` and per-step result schemas  
- `.env.example` documents every variable (no real secrets)  
- `README.md` stub with project title and TODO sections  

**Todo List**  
- [ ] Create all directories (`app/agents`, `app/services`, `app/models`, `app/prompts`, `dashboard`, `tests`)  
- [ ] Write `requirements.txt` (streamlit, plotly, yfinance, pandas, numpy, pydantic, ibm-watsonx-ai, ibm-cos-sdk, feedparser, python-dotenv, requests, fpdf2, pytest, pytest-mock)  
- [ ] Write `app/config.py` — load and expose: `WATSONX_API_KEY`, `WATSONX_URL` (default `https://au-syd.ml.cloud.ibm.com`), `WATSONX_PROJECT_ID`, `GRANITE_MODEL_ID` (default `ibm/granite-13b-instruct-v2`), `COS_API_KEY`, `COS_INSTANCE_CRN`, `COS_BUCKET`, `COS_ENDPOINT`, `NEWS_API_KEY`, `MOCK_MODE` (bool, default `false`)  
- [ ] Write `app/models/state.py` — `AnalysisState` with fields for each step's outputs, `StepStatus` enum (PENDING / RUNNING / DONE / FAILED), list of per-step `StepResult` entries  
- [ ] Write `.env.example`  
- [ ] Write `README.md` stub  

**Relevant Context**  
- watsonx.ai SDK: `ibm_watsonx_ai.foundation_models.ModelInference`  
- COS SDK: `ibm_boto3` with `ibm_botocore`  
- Config pattern: `pydantic-settings` or plain `python-dotenv` + dataclass  

**Status** `[ ] pending`

---

## Sub-Task 2 — Services Layer

**Intent**  
Implement the three reusable service modules that all pipeline steps depend on,
with real implementations and mock fallbacks.

**Expected Outcomes**  
- `app/services/watsonx_client.py` — `WatsonxClient` class with `generate(prompt, max_tokens)` method; uses real SDK when `MOCK_MODE=false`, returns a configurable stub string when `MOCK_MODE=true`  
- `app/services/data_sources.py` — `fetch_price_history(ticker, period)`, `fetch_fundamentals(ticker)`, `fetch_news(ticker)` helpers; mock returns synthetic DataFrames/dicts  
- `app/services/storage.py` — `upload_report(key, content)`, `download_report(key)` for COS; local-file fallback when COS is not configured  

**Todo List**  
- [ ] Write `watsonx_client.py` with real `ModelInference` call + mock branch  
- [ ] Write `data_sources.py` with `yfinance` calls + `feedparser`/`NewsAPI` news fetch + mock data factories  
- [ ] Write `storage.py` with `ibm_boto3` COS client + local `reports/` directory fallback  

**Relevant Context**  
- `ibm_watsonx_ai.foundation_models.ModelInference.generate_text()`  
- `yfinance.Ticker(symbol).history()`, `.info`, `.financials`  
- `ibm_boto3.resource('s3', ...)` with IBM endpoint config  

**Status** `[ ] pending`

---

## Sub-Task 3 — Pipeline Steps 1–5

**Intent**  
Implement the first five deterministic (no-LLM) pipeline steps as standalone
modules, each exposing a single `run(state: AnalysisState) -> AnalysisState`
function.

**Expected Outcomes**  
- **Step 1** `step1_data_gathering.py` — populates `state.price_history`, `state.fundamentals`, `state.news_items`, `state.analyst_opinions`  
- **Step 2** `step2_validation.py` — sets `state.data_quality_score` (0–100), `state.validation_flags` list  
- **Step 3** `step3_kpi_extraction.py` — populates `state.kpis` dict (revenue, net income, EBITDA, margins, EPS, debt, cash flow, YoY/QoQ); falls back to Granite for unstructured extraction  
- **Step 4** `step4_ratios.py` — populates `state.ratios` (P/E, P/B, ROE, ROA, D/E, current ratio, interest coverage, beta, volatility, max drawdown)  
- **Step 5** `step5_benchmarking.py` — populates `state.benchmark_table` (ranked DataFrame vs. peers + sector median), `state.relative_strength`  

**Todo List**  
- [ ] Implement `step1_data_gathering.py` using `data_sources.py`  
- [ ] Implement `step2_validation.py` — null checks, staleness (>3 days), outlier detection (z-score >3), currency flag, dedup news  
- [ ] Implement `step3_kpi_extraction.py` — structured extraction first, Granite fallback for text  
- [ ] Implement `step4_ratios.py` — pure pandas/numpy calculations  
- [ ] Implement `step5_benchmarking.py` — loop peer tickers through steps 1–4, build comparison table  

**Relevant Context**  
- Sector median can be approximated from the peer set itself when no broader benchmark is available  
- Beta calculation: 252-day rolling covariance vs. market index (^NSEI or ^GSPC)  
- Max drawdown: `(cummax - price) / cummax`  

**Status** `[ ] pending`

---

## Sub-Task 4 — Pipeline Steps 6–9 with Granite

**Intent**  
Implement the four AI-powered pipeline steps that call Granite via
`WatsonxClient`, using versioned prompt templates from `app/prompts/`.

**Expected Outcomes**  
- **Step 6** `step6_sentiment.py` — for each news item calls Granite with `prompts/sentiment_classification.txt`; stores per-item sentiment label + confidence; computes `state.sentiment_index` time-series  
- **Step 7** `step7_predictive.py` — moving average trend detection (20/50 day), linear projection for next 30 days, populates `state.risk_indicators` (Low/Medium/High with explanation) for leverage, margin compression, volatility, sentiment spike, valuation  
- **Step 8** `step8_recommendations.py` — calls Granite with `prompts/recommendation.txt`, grounded in `state` data only; parses structured output into `state.recommendation` (rating, confidence, reasons, risks, data_citations)  
- **Step 9** `step9_report.py` — serialises `state` to JSON, renders HTML summary, uploads via `storage.py`; populates `state.report_url`  
- All three Granite prompt templates written and versioned  

**Todo List**  
- [ ] Write `prompts/sentiment_classification.txt` (few-shot, structured JSON output)  
- [ ] Write `prompts/kpi_extraction.txt` (structured JSON output)  
- [ ] Write `prompts/recommendation.txt` (chain-of-thought, JSON output with citations)  
- [ ] Implement `step6_sentiment.py` with batch news processing + aggregation  
- [ ] Implement `step7_predictive.py` — MA crossover, linear regression projection, risk scoring logic  
- [ ] Implement `step8_recommendations.py` — prompt assembly from state, Granite call, JSON parse + fallback  
- [ ] Implement `step9_report.py` — JSON serialisation, HTML template render, COS/local storage  

**Relevant Context**  
- Granite prompt format: instruction-tuned, expects clear role + task + structured output spec  
- Parse Granite JSON output defensively — use `json.loads` with try/except and a regex fallback  
- `state.data_citations` must list only fields that were actually computed (not None)  

**Status** `[ ] pending`

---

## Sub-Task 5 — Orchestrator

**Intent**  
Write `agents/orchestrator.py` which runs all 9 steps in sequence, manages
retries, updates step status in `state.step_results`, and yields live progress
events back to the Streamlit UI via a generator/callback.

**Expected Outcomes**  
- `Orchestrator.run(state, on_progress)` iterates steps 1–9 in order  
- Each step is wrapped: catches exceptions, retries up to 2×, marks FAILED with message on final failure, continues pipeline (graceful degradation)  
- `on_progress(step_number, status, message)` callback fires before and after each step  
- Structured logging (Python `logging`) for every step entry/exit, duration, and errors  
- `state.step_results` list fully populated after run  

**Todo List**  
- [ ] Define `PIPELINE_STEPS` list mapping step number → module function  
- [ ] Implement retry wrapper with exponential back-off (0.5s, 1s)  
- [ ] Implement `on_progress` callback protocol  
- [ ] Add structured log statements (step name, ticker, duration_ms, status)  
- [ ] Handle partial failure: if Step 1 fails fatally, skip Steps 2–9 with SKIPPED status  

**Relevant Context**  
- Use `time.perf_counter()` for duration tracking  
- Streamlit `st.session_state` will hold the live `state` object; the callback updates a separate status list  

**Status** `[ ] pending`

---

## Sub-Task 6 — Streamlit Dashboard

**Intent**  
Build `dashboard/app.py` — the full conversational UI that takes user input,
calls the orchestrator, and renders all result sections.

**Expected Outcomes**  
- **Sidebar:** ticker input (comma-separated), peer tickers, date range slider, "Run Analysis" button  
- **Pipeline progress tracker:** 9-step status list updates live during run  
- **Investment Summary:** 3–4 metric cards (rating, confidence, data quality, sentiment)  
- **Price & Trend chart:** Plotly candlestick + 20/50 MA overlay  
- **KPI/Ratio table:** styled DataFrame  
- **Peer Benchmarking chart:** Plotly grouped bar chart  
- **Sentiment Timeline:** Plotly line chart of sentiment index  
- **Risk Indicators:** colour-coded gauge or heatmap (Low=green, Medium=amber, High=red)  
- **AI Recommendations panel:** formatted text, citations, disclaimer banner  
- **Download button:** link to JSON/HTML report  

**Todo List**  
- [ ] Scaffold `dashboard/app.py` with `st.set_page_config` and session state init  
- [ ] Build sidebar input widgets  
- [ ] Build live pipeline progress tracker (loop over `state.step_results`)  
- [ ] Build Investment Summary section  
- [ ] Build Price & Trend section with Plotly  
- [ ] Build KPI/Ratio table section  
- [ ] Build Peer Benchmarking section  
- [ ] Build Sentiment Timeline section  
- [ ] Build Risk Indicators section  
- [ ] Build AI Recommendations panel + disclaimer  
- [ ] Wire "Run Analysis" button to `Orchestrator.run()` with `st.status` / spinner  

**Relevant Context**  
- Use `st.fragment` or `st.empty` placeholders for live updates  
- Streamlit 1.59 (installed) supports `st.status` for multi-step progress  
- Plotly charts via `st.plotly_chart(fig, use_container_width=True)`  

**Status** `[ ] pending`

---

## Sub-Task 7 — Tests

**Intent**  
Write pytest unit tests for the three most critical, non-trivial modules with
mocked external calls.

**Expected Outcomes**  
- `tests/test_ratios.py` — 5+ tests for ratio calculations (P/E, ROE, max drawdown, beta) with known inputs  
- `tests/test_validation.py` — tests for null detection, staleness flag, duplicate news dedup, quality score range  
- `tests/test_orchestrator.py` — tests that orchestrator runs all steps, handles a step failure gracefully, populates step_results  

**Todo List**  
- [ ] Write `test_ratios.py`  
- [ ] Write `test_validation.py`  
- [ ] Write `test_orchestrator.py` using `pytest-mock` to stub step functions  
- [ ] Verify `pytest` runs clean from project root  

**Relevant Context**  
- Mock `WatsonxClient.generate` to return fixture JSON strings  
- Mock `yfinance.Ticker` to return synthetic DataFrames  

**Status** `[ ] pending`

---

## Sub-Task 8 — Docker & IBM Cloud Deployment

**Intent**  
Produce a production-ready `Dockerfile` and a README section with exact
`ibmcloud` CLI commands to deploy on Code Engine and provision dependencies.

**Expected Outcomes**  
- `Dockerfile` — Python 3.11 slim base, non-root user, `EXPOSE 8501`, `CMD streamlit run dashboard/app.py`  
- `README.md` — complete with architecture diagram (Mermaid), setup, local run, test, and deploy sections  
- Deploy steps cover: COS bucket creation, watsonx.ai project setup, Code Engine app creation with env vars, custom domain (optional)  
- Demo script `demo.py` that runs the pipeline on `TCS.NS` + `INFY.NS` vs. `WIPRO.NS` + `HCL.NS` in mock mode and prints the recommendation  

**Todo List**  
- [ ] Write `Dockerfile`  
- [ ] Write complete `README.md` with Mermaid diagram and all sections  
- [ ] Write `demo.py`  
- [ ] Verify `docker build` succeeds (syntax check)  

**Relevant Context**  
- Code Engine app: `ibmcloud ce app create --name invest-analyst --image <image> --port 8501`  
- COS: `ibmcloud resource service-instance-create invest-cos cloud-object-storage lite global`  
- watsonx.ai: provisioned via IBM Cloud catalog, project created in UI or via API  

**Status** `[ ] pending`

---

## Implementation Order

Sub-tasks are designed to be executed sequentially:

1. Scaffold → 2. Services → 3. Steps 1–5 → 4. Steps 6–9 → 5. Orchestrator → 6. Dashboard → 7. Tests → 8. Docker/Deploy

Each sub-task is independently reviewable before the next begins.
