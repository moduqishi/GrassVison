"""Configuration loading, reloading, atomic write, and backup."""
from __future__ import annotations

import os
import re
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

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
BACKUP_DIR = BASE_DIR / "config" / "backups"
PROMPTS_DIR = BASE_DIR / "config" / "prompts"
MAX_BACKUPS = 10

_config: AppConfig | None = None
_config_loaded_at: datetime | None = None
_config_error: str | None = None

# Regex to resolve ${VAR} or ${VAR:default} references
_VAR_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} and ${ENV_VAR:default} in strings."""
    if isinstance(value, str):
        def _replace(m):
            var = m.group(1)
            default = m.group(2)
            return os.environ.get(var, default if default is not None else m.group(0))
        return _VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


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

    raw = _resolve_env(raw)
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
    """Atomically save config: backup → write .tmp → validate → replace."""
    # Dump model to dict (with aliases preserved)
    raw = config.model_dump(exclude_none=False, mode="python")

    # Resolve env vars back to placeholder form for clean YAML
    # We keep the raw dict as-is since users may have literal keys
    yaml_str = yaml.dump(raw, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Create backup
    backup_config()

    # Write to temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=CONFIG_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        # Validate by loading
        with open(tmp_path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)

        # Atomic replace
        os.replace(tmp_path, CONFIG_PATH)
    except Exception:
        # Clean up temp on failure
        Path(tmp_path).unlink(missing_ok=True)
        raise ConfigError("Failed to write config: validation error. Original config preserved.")


def read_yaml_file(path: str) -> dict:
    """Read a YAML file and return parsed dict."""
    fp = BASE_DIR / path if not path.startswith("/") else Path(path)
    if not fp.exists():
        raise ConfigError(f"File not found: {fp}")
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_prompt(name: str) -> str:
    """Read a prompt file from config/prompts/."""
    fp = PROMPTS_DIR / name
    if not fp.exists():
        raise ConfigError(f"Prompt not found: {name}")
    return fp.read_text(encoding="utf-8")


def write_prompt(name: str, content: str) -> None:
    """Write a prompt file."""
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
    """List all prompt file names."""
    if not PROMPTS_DIR.exists():
        return []
    return sorted(
        [f.name for f in PROMPTS_DIR.iterdir() if f.is_file() and f.suffix == ".txt"]
    )
