"""Tests for image cache and preprocessing."""
import asyncio
import pytest
from app.image_cache import ImageCache, CacheEntry, get_image_cache, _now


@pytest.fixture
def cache():
    return ImageCache(enabled=True, ttl_seconds=10, max_entries=5)


@pytest.mark.asyncio
async def test_get_or_reserve_owner(cache):
    entry, status = await cache.get_or_reserve("key1")
    assert entry is None
    assert status == "owner"


@pytest.mark.asyncio
async def test_set_and_get_cached(cache):
    entry, status = await cache.get_or_reserve("key1")
    assert status == "owner"

    now = _now()
    await cache.set("key1", CacheEntry(
        result="test result",
        content_hash="abc",
        provider_id="p1",
        model_id="m1",
        prompt_hash="ph",
        analysis_mode="independent",
        created_at=now,
        expires_at=now + 10,
    ))

    entry2, status2 = await cache.get_or_reserve("key1")
    assert status2 == "cached"
    assert entry2.result == "test result"


@pytest.mark.asyncio
async def test_ttl_expiry(cache):
    now = _now()
    await cache.set("key1", CacheEntry(
        result="old", content_hash="abc",
        provider_id="p1", model_id="m1", prompt_hash="ph",
        analysis_mode="independent",
        created_at=now - 20,
        expires_at=now - 5,  # already expired
    ))

    entry, status = await cache.get_or_reserve("key1")
    assert status == "owner"  # expired → miss → new owner


@pytest.mark.asyncio
async def test_lru_eviction(cache):
    now = _now()
    for i in range(6):  # max_entries=5
        await cache.set(f"key{i}", CacheEntry(
            result=f"r{i}", content_hash=f"h{i}",
            provider_id="p", model_id="m", prompt_hash="ph",
            analysis_mode="independent",
            created_at=now + i,
            expires_at=now + 100,
        ))
    # key0 should be evicted (oldest)
    entry, status = await cache.get_or_reserve("key0")
    assert status == "owner"  # not found


@pytest.mark.asyncio
async def test_move_to_end_on_hit(cache):
    now = _now()
    for i in range(5):
        await cache.set(f"key{i}", CacheEntry(
            result=f"r{i}", content_hash=f"h{i}",
            provider_id="p", model_id="m", prompt_hash="ph",
            analysis_mode="independent",
            created_at=now + i,
            expires_at=now + 100,
        ))
    # Access key0 to make it MRU
    await cache.get_or_reserve("key0")
    # Add key5 → should evict key1 (now oldest)
    await cache.set("key5", CacheEntry(
        result="r5", content_hash="h5",
        provider_id="p", model_id="m", prompt_hash="ph",
        analysis_mode="independent",
        created_at=now + 10, expires_at=now + 100,
    ))
    entry0, s0 = await cache.get_or_reserve("key0")
    assert s0 == "cached"  # key0 survived
    entry1, s1 = await cache.get_or_reserve("key1")
    assert s1 == "owner"  # key1 was evicted


@pytest.mark.asyncio
async def test_reconfigure_disable(cache):
    await cache.reconfigure(enabled=False)
    entry, status = await cache.get_or_reserve("any")
    assert status == "owner"
    s = await cache.stats()
    assert s["enabled"] is False


@pytest.mark.asyncio
async def test_reconfigure_ttl(cache):
    now = _now()
    await cache.set("key1", CacheEntry(
        result="test", content_hash="abc",
        provider_id="p", model_id="m", prompt_hash="ph",
        analysis_mode="independent",
        created_at=now - 100,
        expires_at=now + 1000,  # far in future
    ))
    # Change TTL to 5s → entry should expire
    await cache.reconfigure(ttl_seconds=5)
    entry, status = await cache.get_or_reserve("key1")
    assert status == "owner"  # expired after TTL change


@pytest.mark.asyncio
async def test_stats(cache):
    await cache.set("k1", CacheEntry(
        result="r1", content_hash="h1",
        provider_id="p", model_id="m", prompt_hash="ph",
        analysis_mode="independent",
        created_at=_now(), expires_at=_now() + 100,
    ))
    await cache.get_or_reserve("k1")  # hit
    await cache.get_or_reserve("k2")  # miss
    s = await cache.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["size"] == 1
    assert s["vision_calls_saved"] == 1
