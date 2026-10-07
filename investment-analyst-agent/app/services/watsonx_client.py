"""
app/services/watsonx_client.py — thin wrapper around ibm-watsonx-ai SDK.
Falls back to a configurable stub when MOCK_MODE=true or credentials are absent.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ── Mock stubs keyed by prompt type keyword ───────────────────────────────────
_MOCK_RESPONSES: dict[str, str] = {
    "sentiment": json.dumps({
        "sentiment": "positive",
        "confidence": 0.82,
        "rationale": "Article highlights strong revenue growth and positive guidance."
    }),
    "kpi": json.dumps({
        "revenue": 2500000000,
        "net_income": 350000000,
        "ebitda": 520000000,
        "operating_margin": 0.21,
        "net_margin": 0.14,
        "eps": 12.5,
        "total_debt": 800000000,
        "free_cash_flow": 290000000,
        "revenue_yoy": 0.12,
        "net_income_yoy": 0.08
    }),
    "recommendation": json.dumps({
        "rating": "Buy",
        "confidence": 0.76,
        "summary": "Strong fundamentals with healthy margins and low leverage support a Buy rating.",
        "reasons": [
            "Revenue grew 12% YoY with expanding margins",
            "Low debt-to-equity ratio of 0.42",
            "Positive analyst consensus and sentiment"
        ],
        "risks": [
            "Global macro headwinds could pressure margins",
            "Valuation is slightly elevated vs peers"
        ],
        "citations": ["revenue_yoy", "net_margin", "debt_to_equity", "overall_sentiment"]
    }),
}


def _pick_mock(prompt: str) -> str:
    pl = prompt.lower()
    if "sentiment" in pl:
        return _MOCK_RESPONSES["sentiment"]
    if "kpi" in pl or "revenue" in pl or "extract" in pl:
        return _MOCK_RESPONSES["kpi"]
    return _MOCK_RESPONSES["recommendation"]


class WatsonxClient:
    """
    Wraps IBM watsonx.ai ModelInference for text generation.
    Automatically uses mock mode when settings.mock_mode is True
    or when watsonx credentials are not configured.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._use_mock = settings.mock_mode or not settings.watsonx_configured

        if not self._use_mock:
            self._init_real_client()

    def _init_real_client(self) -> None:
        try:
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

            credentials = Credentials(
                url=settings.watsonx_url,
                api_key=settings.watsonx_api_key,
            )
            self._gen_params = {
                GenParams.MAX_NEW_TOKENS: 1024,
                GenParams.TEMPERATURE: 0.1,
                GenParams.STOP_SEQUENCES: ["```"],
            }
            self._model = ModelInference(
                model_id=settings.granite_model_id,
                credentials=credentials,
                project_id=settings.watsonx_project_id,
                params=self._gen_params,
            )
            logger.info(
                "WatsonxClient initialised | model=%s url=%s",
                settings.granite_model_id,
                settings.watsonx_url,
            )
        except Exception as exc:
            logger.warning(
                "WatsonxClient: could not initialise real client (%s) — switching to mock.",
                exc,
            )
            self._use_mock = True

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Generate text from a prompt.  Returns the raw string response.
        Raises RuntimeError only on hard failures in non-mock mode.
        """
        if self._use_mock:
            logger.debug("WatsonxClient [MOCK] generating response.")
            return _pick_mock(prompt)

        try:
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

            params = {**self._gen_params, GenParams.MAX_NEW_TOKENS: max_tokens}
            response = self._model.generate_text(prompt=prompt, params=params)
            # SDK returns a string directly from generate_text
            return response if isinstance(response, str) else str(response)
        except Exception as exc:
            logger.error("WatsonxClient.generate failed: %s", exc)
            # Graceful degradation — return mock so pipeline can continue
            return _pick_mock(prompt)

    def extract_json(self, prompt: str, max_tokens: int = 1024) -> dict:
        """
        Generate text and attempt to parse JSON from the response.
        Returns an empty dict on parse failure (caller handles missing keys).
        """
        raw = self.generate(prompt, max_tokens)
        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON block from markdown fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning("WatsonxClient.extract_json: could not parse JSON from response.")
        return {}


# Module-level singleton
_client: WatsonxClient | None = None


def get_watsonx_client() -> WatsonxClient:
    global _client
    if _client is None:
        _client = WatsonxClient()
    return _client
