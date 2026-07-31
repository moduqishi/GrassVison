"""Configuration loading, reloading, atomic write, and backup."""
from __future__ import annotations

import logging
import os
import sys
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from app.errors import ConfigError
from app.schemas import (
    AppConfig,
    ServerConfig,
    AdminConfig,
    SourceProviderConfig,
    VisionProviderConfig,
    EnhancedModelConfig,
    ImageConfig,
    LoggingConfig,
)

logger = logging.getLogger(__name__)

# ── Path resolution: frozen (PyInstaller) vs dev ──────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent.resolve()
    BUNDLE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    BUNDLE_DIR = BASE_DIR

CONFIG_PATH = BASE_DIR / "config.yaml"
BACKUP_DIR = BASE_DIR / "config" / "backups"
PROMPTS_DIR = BASE_DIR / "config" / "prompts"
STATS_DIR = BASE_DIR / "stats"
LOGS_DIR = BASE_DIR / "logs"
MAX_BACKUPS = 10

_config: AppConfig | None = None
_config_loaded_at: datetime | None = None
_config_error: str | None = None


def load_config(reload: bool = False) -> AppConfig:
    """Load and validate YAML config.  Raises ConfigError on failure."""
    global _config, _config_loaded_at, _config_error

    load_dotenv(BASE_DIR / ".env", override=False)

    if not CONFIG_PATH.exists():
        raise ConfigError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None or not isinstance(raw, dict):
        raise ConfigError("Config file is empty or invalid YAML")

    # Migrate legacy disable_thinking field to extra_params
    vision_providers = raw.get("vision_providers")
    if isinstance(vision_providers, dict):
        for key, provider in vision_providers.items():
            if not isinstance(provider, dict) or "disable_thinking" not in provider:
                continue
            disable_thinking = provider.pop("disable_thinking", None)
            extra_params = provider.get("extra_params")
            if disable_thinking is True:
                if not extra_params:
                    provider["extra_params"] = {"enable_thinking": False, "thinking": False}
                    logger.warning(
                        f"Vision provider '{key}': migrated disable_thinking=true to "
                        f"extra_params={{enable_thinking: false, thinking: false}}; "
                        f"please update config.yaml manually"
                    )
                else:
                    logger.warning(
                        f"Vision provider '{key}': both disable_thinking and extra_params "
                        f"present; keeping extra_params and ignoring disable_thinking"
                    )
            else:
                logger.info(
                    f"Vision provider '{key}': disable_thinking=false is now the default, "
                    f"field ignored"
                )

    _config = AppConfig(**raw)
    _config_loaded_at = datetime.now()
    _config_error = None
    return _config


def get_config() -> AppConfig:
    """Return the current config.  Lazily loads on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> AppConfig:
    """Force reload from disk."""
    return load_config(reload=True)


def get_config_meta() -> dict:
    return {
        "loaded_at": _config_loaded_at,
        "error": _config_error,
        "path": str(CONFIG_PATH),
    }


def _prune_backups():
    backups = sorted(BACKUP_DIR.glob("config-*.yaml"))
    for old in backups[:-MAX_BACKUPS]:
        old.unlink()


def backup_config() -> Path:
    """Create a timestamped backup of the current config file."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = BACKUP_DIR / f"config-{ts}.yaml"
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, dest)
    _prune_backups()
    return dest


def save_config(config: AppConfig) -> None:
    """Save config: backup -> write .tmp -> validate -> copy to target.

    用 shutil.copy2 而非 os.replace，以保留目标文件的 inode。
    这是 Docker 单文件 bind mount (`./config.yaml:/app/config.yaml`) 的必要妥协：
    os.replace 会换掉 inode，宿主机文件不会同步更新，导致保存「只在内存里」、容器重启后丢失。
    copy2 保留原 inode，bind mount 同步正常；原子性由先写临时文件 + 校验 + 备份兜底。
    """
    raw = config.model_dump(exclude_none=False, mode="python")
    yaml_str = yaml.dump(raw, default_flow_style=False, allow_unicode=True, sort_keys=False)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_config()

    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=CONFIG_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        with open(tmp_path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)

        shutil.copy2(tmp_path, CONFIG_PATH)
    except Exception:
        raise ConfigError("Failed to write config: validation error. Original config preserved.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def read_yaml_file(path: str) -> dict:
    """Read a YAML file and return parsed dict."""
    fp = BASE_DIR / path if not path.startswith("/") else Path(path)
    if not fp.exists():
        raise ConfigError(f"File not found: {fp}")
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_prompt(name: str) -> str:
    """Read a prompt file from config/prompts/ (falls back to bundle defaults)."""
    fp = PROMPTS_DIR / name
    if fp.exists():
        return fp.read_text(encoding="utf-8")
    bundled = BUNDLE_DIR / "config" / "prompts" / name
    if bundled.exists():
        return bundled.read_text(encoding="utf-8")
    raise ConfigError(f"Prompt not found: {name}")


def write_prompt(name: str, content: str) -> None:
    """Write a prompt file to the user data directory."""
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    fp = PROMPTS_DIR / name
    fp.write_text(content, encoding="utf-8")


def delete_prompt(name: str) -> None:
    """Delete a prompt file.  Refuses deletion of 'default.txt'."""
    if name == "default.txt":
        raise ConfigError("Cannot delete default.txt")
    fp = PROMPTS_DIR / name
    if fp.exists():
        fp.unlink()


def list_prompts() -> list[str]:
    """List all prompt file names (merged from bundle defaults + user dir)."""
    seen: set[str] = set()
    result: list[str] = []

    bundle_prompts = BUNDLE_DIR / "config" / "prompts"
    if bundle_prompts.exists():
        for f in sorted(bundle_prompts.iterdir()):
            if f.is_file() and f.suffix == ".txt" and f.name not in seen:
                seen.add(f.name)
                result.append(f.name)

    if PROMPTS_DIR.exists():
        for f in sorted(PROMPTS_DIR.iterdir()):
            if f.is_file() and f.suffix == ".txt" and f.name not in seen:
                seen.add(f.name)
                result.append(f.name)

    return result
