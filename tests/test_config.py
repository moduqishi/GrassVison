"""Tests for configuration loading and atomic write."""
import os
import tempfile
from pathlib import Path

import pytest
from app.config import load_config, get_config, reload_config, _resolve_env


class TestEnvResolution:
    def test_resolves_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "my-secret")
        result = _resolve_env("${TEST_KEY}")
        assert result == "my-secret"

    def test_resolves_with_default(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT", raising=False)
        result = _resolve_env("${NONEXISTENT:fallback}")
        assert result == "fallback"

    def test_resolves_nested_dict(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "sk-123")
        result = _resolve_env({"a": {"b": "${API_KEY}"}})
        assert result["a"]["b"] == "sk-123"


class TestConfigLoading:
    def test_loads_default_config(self):
        # Point config to our test file
        from app import config as cfg_mod
        cfg = cfg_mod.get_config()
        assert cfg is not None
        assert cfg.server.port == 8042
        assert cfg.server.host == "127.0.0.1"
        assert len(cfg.models) >= 1
        assert "deepseek-vision" in cfg.models


class TestModelValidation:
    def test_deepseek_vision_model(self):
        from app.config import get_config
        cfg = get_config()
        model = cfg.models.get("deepseek-vision")
        assert model is not None
        assert model.enabled is True
        assert model.vision_enabled is True
        assert model.source_model == "deepseek-chat"
        assert model.vision_failure_mode == "error"
