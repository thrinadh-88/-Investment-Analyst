"""
app/config.py — centralised settings loaded from environment / .env file.
All pipeline modules import from here; no module reads os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)


def _bool(val: str | None) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # ── IBM watsonx.ai ────────────────────────────────────────────────────────
    watsonx_api_key: str = field(default_factory=lambda: os.getenv("WATSONX_API_KEY", ""))
    watsonx_url: str = field(
        default_factory=lambda: os.getenv("WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")
    )
    watsonx_project_id: str = field(default_factory=lambda: os.getenv("WATSONX_PROJECT_ID", ""))
    granite_model_id: str = field(
        default_factory=lambda: os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")
    )

    # ── IBM Cloud Object Storage ───────────────────────────────────────────────
    cos_api_key: str = field(default_factory=lambda: os.getenv("COS_API_KEY", ""))
    cos_instance_crn: str = field(default_factory=lambda: os.getenv("COS_INSTANCE_CRN", ""))
    cos_bucket: str = field(
        default_factory=lambda: os.getenv("COS_BUCKET", "investment-analyst-reports")
    )
    cos_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "COS_ENDPOINT",
            "https://s3.au-syd.cloud-object-storage.appdomain.cloud",
        )
    )

    # ── News API ──────────────────────────────────────────────────────────────
    news_api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))

    # ── Behaviour flags ───────────────────────────────────────────────────────
    mock_mode: bool = field(default_factory=lambda: _bool(os.getenv("MOCK_MODE", "false")))

    # ── Local fallback directory for reports ──────────────────────────────────
    reports_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "reports"
    )

    @property
    def cos_configured(self) -> bool:
        return bool(self.cos_api_key and self.cos_instance_crn and self.cos_bucket)

    @property
    def watsonx_configured(self) -> bool:
        return bool(self.watsonx_api_key and self.watsonx_project_id)


settings = Settings()
