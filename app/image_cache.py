"""Image hash cache with LRU eviction, TTL, and single-flight deduplication."""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    result: str
    content_hash: str
    provider_id: str
    model_id: str
    prompt_hash: str
    analysis_mode: str
    created_at: float
    expires_at: float


def _now() -> float:
    return time.monotonic()


class ImageCache:
    def __init__(self, enabled: bool = True, ttl_seconds: int = 3600, max_entries: int = 200):
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._lock = asyncio.Lock()
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._inflight: dict[str, asyncio.Future] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0
        self._calls_saved = 0

    # ── Core API ─────────────────────────────────────────────────

    async def get_or_reserve(self, cache_key: str) -> tuple[CacheEntry | None, str]:
        """
        Returns (entry, "cached") | (None, "owner") | (None, "waiter").

        "owner" — caller must perform analysis and call set().
        "waiter" — caller must await the inflight Future.
        """
        if not self.enabled:
            return (None, "owner")

        async with self._lock:
            # Check cache (with TTL expiry)
            entry = self._store.get(cache_key)
            if entry is not None:
                if entry.expires_at <= _now():
                    del self._store[cache_key]
                    self._expired += 1
                    self._misses += 1
                    # fall through to inflight / owner
                else:
                    self._store.move_to_end(cache_key)
                    self._hits += 1
                    self._calls_saved += 1
                    return (entry, "cached")

            # Check inflight
            if cache_key in self._inflight:
                fut = self._inflight[cache_key]
                self._misses += 1
                return (None, "waiter")

            # Become owner
            self._misses += 1
            fut = asyncio.get_event_loop().create_future()
            self._inflight[cache_key] = fut
            return (None, "owner")

    async def set(self, cache_key: str, entry: CacheEntry) -> None:
        if not self.enabled:
            return
        async with self._lock:
            # Resolve inflight
            fut = self._inflight.pop(cache_key, None)

            # Evict if at capacity
            while len(self._store) >= self.max_entries:
                self._store.popitem(last=False)
                self._evictions += 1

            self._store[cache_key] = entry
            self._store.move_to_end(cache_key)

        # Notify waiters outside the lock
        if fut and not fut.done():
            fut.set_result(entry)

    async def wait_inflight(self, cache_key: str) -> CacheEntry:
        async with self._lock:
            fut = self._inflight.get(cache_key)
        if fut:
            return await fut
        # Fallback — shouldn't happen normally
        async with self._lock:
            entry = self._store.get(cache_key)
            if entry:
                return entry
            raise RuntimeError(f"Cache key {cache_key[:16]}... not found")

    async def invalidate(self, cache_key: str) -> bool:
        async with self._lock:
            if cache_key in self._store:
                del self._store[cache_key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._inflight.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expired = 0
            self._calls_saved = 0

    # ── Reconfigure ──────────────────────────────────────────────

    async def reconfigure(self, enabled: bool | None = None, ttl_seconds: int | None = None, max_entries: int | None = None) -> None:
        async with self._lock:
            if enabled is not None and enabled != self.enabled:
                self.enabled = enabled
                if not enabled:
                    self._store.clear()
                    self._inflight.clear()
                    self._hits = 0
                    self._misses = 0
                    self._evictions = 0
                    self._expired = 0
                    self._calls_saved = 0
                return

            if ttl_seconds is not None and ttl_seconds != self.ttl_seconds:
                old_ttl = self.ttl_seconds
                self.ttl_seconds = ttl_seconds
                now = _now()
                expired_keys = []
                for key in list(self._store.keys()):
                    entry = self._store[key]
                    # Recalculate expires_at from created_at + new_ttl
                    entry.expires_at = entry.created_at + ttl_seconds
                    if entry.expires_at <= now:
                        expired_keys.append(key)
                for key in expired_keys:
                    del self._store[key]
                    self._expired += 1

            if max_entries is not None and max_entries != self.max_entries:
                self.max_entries = max_entries
                while len(self._store) > self.max_entries:
                    self._store.popitem(last=False)
                    self._evictions += 1

    # ── Stats ────────────────────────────────────────────────────

    async def stats(self) -> dict:
        async with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total * 100, 1) if total else 0,
                "size": len(self._store),
                "max_entries": self.max_entries,
                "evictions": self._evictions,
                "expired": self._expired,
                "vision_calls_saved": self._calls_saved,
                "ttl_seconds": self.ttl_seconds,
                "enabled": self.enabled,
            }


# Global singleton
_cache: ImageCache | None = None


def get_image_cache() -> ImageCache:
    global _cache
    if _cache is None:
        from app.config import get_config
        cfg = get_config().image.vision_cache
        _cache = ImageCache(
            enabled=cfg.enabled,
            ttl_seconds=cfg.ttl_seconds,
            max_entries=cfg.max_entries,
        )
    return _cache
