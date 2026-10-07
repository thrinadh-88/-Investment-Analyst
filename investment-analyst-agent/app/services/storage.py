"""
app/services/storage.py — IBM Cloud Object Storage upload/download with
automatic fallback to local reports/ directory.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def _ensure_local_dir() -> Path:
    d = settings.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def upload_report(key: str, content: str | bytes, content_type: str = "application/json") -> str:
    """
    Upload content to COS or save locally.
    Returns the public-accessible URL (COS) or local file path.
    """
    if settings.cos_configured and not settings.mock_mode:
        try:
            import ibm_boto3
            from ibm_botocore.client import Config

            cos = ibm_boto3.client(
                "s3",
                ibm_api_key_id=settings.cos_api_key,
                ibm_service_instance_id=settings.cos_instance_crn,
                config=Config(signature_version="oauth"),
                endpoint_url=settings.cos_endpoint,
            )
            body = content.encode() if isinstance(content, str) else content
            cos.put_object(
                Bucket=settings.cos_bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
            url = f"{settings.cos_endpoint}/{settings.cos_bucket}/{key}"
            logger.info("upload_report: uploaded to COS — %s", url)
            return url
        except Exception as exc:
            logger.warning("COS upload failed (%s) — falling back to local.", exc)

    # Local fallback
    local_path = _ensure_local_dir() / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(local_path, mode, encoding=None if isinstance(content, bytes) else "utf-8") as f:
        f.write(content)
    logger.info("upload_report: saved locally — %s", local_path)
    return str(local_path)


def download_report(key: str) -> str | None:
    """Download content from COS or read from local fallback. Returns string or None."""
    if settings.cos_configured and not settings.mock_mode:
        try:
            import ibm_boto3
            from ibm_botocore.client import Config

            cos = ibm_boto3.client(
                "s3",
                ibm_api_key_id=settings.cos_api_key,
                ibm_service_instance_id=settings.cos_instance_crn,
                config=Config(signature_version="oauth"),
                endpoint_url=settings.cos_endpoint,
            )
            obj = cos.get_object(Bucket=settings.cos_bucket, Key=key)
            return obj["Body"].read().decode("utf-8")
        except Exception as exc:
            logger.warning("COS download failed (%s) — trying local.", exc)

    local_path = _ensure_local_dir() / key
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")
    return None


def list_reports() -> list[str]:
    """List available report keys (local only for now)."""
    d = _ensure_local_dir()
    return [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()]
