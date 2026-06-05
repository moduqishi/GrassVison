"""HTTPX async client management for source and vision providers."""
from __future__ import annotations

import httpx

from app.config import get_config
from app.schemas import SourceProviderConfig, VisionProviderConfig


def _merge_headers(base: dict[str, str], api_key: str) -> dict[str, str]:
    h = {**base}
    if api_key:
        h.setdefault("Authorization", f"Bearer {api_key}")
    return h


def get_source_client(provider: SourceProviderConfig) -> httpx.AsyncClient:
    """Build an httpx client for a source (text) provider."""
    return httpx.AsyncClient(
        base_url=provider.base_url.rstrip("/"),
        headers=_merge_headers(provider.headers, provider.api_key),
        timeout=provider.timeout,
    )


def get_vision_client(provider: VisionProviderConfig) -> httpx.AsyncClient:
    """Build an httpx client for a vision provider."""
    return httpx.AsyncClient(
        base_url=provider.base_url.rstrip("/"),
        headers=_merge_headers(provider.headers, provider.api_key),
        timeout=provider.timeout,
    )


async def test_source_connection(provider: SourceProviderConfig) -> dict:
    """Quick connectivity test for a source provider."""
    client = get_source_client(provider)
    try:
        resp = await client.get("/models", timeout=10)
        status = resp.status_code
        body = resp.text[:500]
        return {"ok": status < 500, "status": status, "body": body}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        await client.aclose()


async def test_vision_connection(provider: VisionProviderConfig) -> dict:
    """Quick connectivity test for a vision provider."""
    client = get_vision_client(provider)
    try:
        resp = await client.get("/models", timeout=10)
        status = resp.status_code
        body = resp.text[:500]
        return {"ok": status < 500, "status": status, "body": body}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        await client.aclose()
