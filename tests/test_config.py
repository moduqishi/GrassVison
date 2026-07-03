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


class TestDisableThinkingMigration:
    """Tests for migrating legacy disable_thinking field to extra_params."""

    def _write_config(self, tmp_path, vision_provider_yaml: str):
        cfg_content = f"""
server:
  host: 127.0.0.1
  port: 8042
admin:
  enabled: false
vision_providers:
  test:
    name: test
    base_url: http://example.com
    api_key: sk-test
    model: test-model
{vision_provider_yaml}
models: {{}}
"""
        fake = tmp_path / "config.yaml"
        fake.write_text(cfg_content, encoding="utf-8")
        return fake

    def test_migrate_disable_thinking_true(self, tmp_path, monkeypatch):
        import app.config as cfgmod

        fake = self._write_config(tmp_path, "    disable_thinking: true")
        monkeypatch.setattr(cfgmod, "CONFIG_PATH", fake)
        monkeypatch.setattr(cfgmod, "_config", None)
        cfg = load_config()
        assert cfg.vision_providers["test"].extra_params == {
            "enable_thinking": False,
            "thinking": False,
        }

    def test_migrate_disable_thinking_false(self, tmp_path, monkeypatch):
        import app.config as cfgmod

        fake = self._write_config(tmp_path, "    disable_thinking: false")
        monkeypatch.setattr(cfgmod, "CONFIG_PATH", fake)
        monkeypatch.setattr(cfgmod, "_config", None)
        cfg = load_config()
        assert cfg.vision_providers["test"].extra_params == {}

    def test_migrate_disable_thinking_true_with_extra_params(self, tmp_path, monkeypatch):
        import app.config as cfgmod

        fake = self._write_config(
            tmp_path,
            "    disable_thinking: true\n    extra_params:\n      reasoning_effort: none",
        )
        monkeypatch.setattr(cfgmod, "CONFIG_PATH", fake)
        monkeypatch.setattr(cfgmod, "_config", None)
        cfg = load_config()
        assert cfg.vision_providers["test"].extra_params == {"reasoning_effort": "none"}

    def test_extra_params_only(self, tmp_path, monkeypatch):
        import app.config as cfgmod

        fake = self._write_config(
            tmp_path,
            "    extra_params:\n      foo: bar",
        )
        monkeypatch.setattr(cfgmod, "CONFIG_PATH", fake)
        monkeypatch.setattr(cfgmod, "_config", None)
        cfg = load_config()
        assert cfg.vision_providers["test"].extra_params == {"foo": "bar"}
