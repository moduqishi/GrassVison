"""Tests for the pooled HTTPX client registry."""
import asyncio
from app.providers import get_vision_client, get_source_client, clear_client_pool
from app.schemas import VisionProviderConfig, SourceProviderConfig


class TestClientPool:
    def test_same_fingerprint_reuses_client(self):
        p = VisionProviderConfig(name="x", base_url="https://a.example.com/v1", api_key="k", timeout=30)
        c1 = get_vision_client(p)
        c2 = get_vision_client(p)
        assert c1 is c2, "相同指纹应复用同一 client"

    def test_different_key_gets_different_client(self):
        p1 = VisionProviderConfig(name="x", base_url="https://a.example.com/v1", api_key="k1", timeout=30)
        p2 = VisionProviderConfig(name="x", base_url="https://a.example.com/v1", api_key="k2", timeout=30)
        assert get_vision_client(p1) is not get_vision_client(p2)

    def test_source_and_vision_pools(self):
        s = SourceProviderConfig(name="s", base_url="https://s.example.com/v1", api_key="k", timeout=30)
        v = VisionProviderConfig(name="v", base_url="https://v.example.com/v1", api_key="k", timeout=30)
        assert get_source_client(s) is get_source_client(s)
        assert get_vision_client(v) is get_vision_client(v)
        assert get_source_client(s) is not get_vision_client(v)

    def test_clear_pool(self):
        clear_client_pool()
        p = VisionProviderConfig(name="x", base_url="https://c.example.com/v1", api_key="k", timeout=30)
        c1 = get_vision_client(p)
        clear_client_pool()
        c2 = get_vision_client(p)
        assert c1 is not c2
