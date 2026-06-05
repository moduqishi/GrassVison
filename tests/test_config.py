"""Tests for configuration loading and atomic write."""
import os
import tempfile
from pathlib import Path

import pytest
from app.config import load_config, get_config, reload_config


class TestConfigLoading:
    def test_loads_default_config(self):
        cfg = get_config()
        assert cfg is not None
        assert cfg.server.port == 8042
        assert cfg.server.host == "127.0.0.1"
        assert len(cfg.models) >= 1

    def test_config_is_self_contained(self):
        """config.yaml should not contain ${VAR} references."""
        from app.config import CONFIG_PATH
        text = CONFIG_PATH.read_text(encoding="utf-8")
        assert "${" not in text, "config.yaml should be self-contained, no env var references"

    def test_backup_and_save(self, tmp_path):
        from app.config import backup_config, save_config, CONFIG_PATH, _config
        from app.schemas import AppConfig

        backup = backup_config()
        assert backup.exists()
        assert backup.name.startswith("config-")
        assert backup.suffix == ".yaml"

        # save with a dummy config to test atomic write
        cfg = get_config()
        cfg.server.port = 9999
        save_config(cfg)
        assert CONFIG_PATH.exists()

        # reload and verify
        reload_config()
        cfg2 = get_config()
        assert cfg2.server.port == 9999

        # restore
        cfg2.server.port = 8042
        save_config(cfg2)


class TestModelValidation:
    def test_default_model_exists(self):
        cfg = get_config()
        models = list(cfg.models.values())
        assert len(models) >= 1
        model = models[0]
        assert model.enabled is True
        assert model.vision_enabled is True
        assert model.replace_response_model is True
        assert model.vision_failure_mode == "error"


class TestConfigReload:
    def test_reload_returns_fresh_config(self):
        cfg1 = get_config()
        cfg2 = reload_config()
        assert cfg1 is not cfg2
        assert cfg2.server.port == 8042
