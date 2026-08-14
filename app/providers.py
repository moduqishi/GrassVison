"""HTTPX async client management with a process-wide connection pool.

每个渠道（base_url+key+timeout+headers 指纹）复用同一个 AsyncClient，
避免每次调用重建连接（TCP+TLS 握手）。配置变化或 TTL 过期时后台关闭旧 client。
"""
from __future__ import annotations

import asyncio
import threading
import time

import httpx

from app.schemas import SourceProviderConfig, VisionProviderConfig

_POOL_TTL_SECONDS = 300  # 池内 client 无使用超过该时长后重建（后台关闭旧连接）
_pool: dict[tuple, tuple[float, httpx.AsyncClient]] = {}
_pool_lock = threading.Lock()


def _fingerprint(base_url: str, api_key: str, timeout: int, headers: dict) -> tuple:
    return (
        base_url.rstrip("/"),
        api_key,
        timeout,
        tuple(sorted((headers or {}).items())),
    )


def _pooled_client(fingerprint: tuple, builder) -> httpx.AsyncClient:
    """按指纹取池化 client；不存在/过期则新建，旧 client 后台关闭。"""
    now = time.monotonic()
    with _pool_lock:
        entry = _pool.get(fingerprint)
        if entry is not None and now - entry[0] < _POOL_TTL_SECONDS:
            return entry[1]
        old = entry[1] if entry is not None else None
        client = builder()
        _pool[fingerprint] = (now, client)
    if old is not None:
        _schedule_close(old)
    return client


def _schedule_close(client: httpx.AsyncClient) -> None:
    """在运行中的事件循环里后台关闭旧 client；无运行循环则交给 GC。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        loop.create_task(client.aclose())
    except RuntimeError:
        pass


def clear_client_pool() -> None:
    """关闭并清空连接池（配置重载/服务关闭时调用）。"""
    with _pool_lock:
        entries = list(_pool.values())
        _pool.clear()
    for _, client in entries:
        _schedule_close(client)


def _merge_headers(base: dict[str, str], api_key: str) -> dict[str, str]:
    h = {**base}
    if api_key:
        h.setdefault("Authorization", f"Bearer {api_key}")
    return h


def get_source_client(provider: SourceProviderConfig) -> httpx.AsyncClient:
    """Get a pooled httpx client for a source (text) provider."""
    fp = _fingerprint(provider.base_url, provider.api_key, provider.timeout, provider.headers)

    def _build() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=provider.base_url.rstrip("/"),
            headers=_merge_headers(provider.headers, provider.api_key),
            timeout=provider.timeout,
        )

    return _pooled_client(fp, _build)


def get_vision_client(provider: VisionProviderConfig) -> httpx.AsyncClient:
    """Get a pooled httpx client for a vision provider."""
    fp = _fingerprint(provider.base_url, provider.api_key, provider.timeout, provider.headers)

    def _build() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=provider.base_url.rstrip("/"),
            headers=_merge_headers(provider.headers, provider.api_key),
            timeout=provider.timeout,
        )

    return _pooled_client(fp, _build)


def get_download_client(timeout: int) -> httpx.AsyncClient:
    """Get a pooled generic client for downloading remote images."""
    fp = ("__download__", timeout)

    def _build() -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout)

    return _pooled_client(fp, _build)


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
