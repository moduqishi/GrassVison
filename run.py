"""GrassVision standalone entry point — used directly or via PyInstaller."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_first_run():
    """On first launch (frozen bundle), create config.yaml & data dirs if missing."""
    if not getattr(sys, "frozen", False):
        return  # dev mode — user runs cp config.example.yaml config.yaml manually

    exe_dir = Path(sys.executable).parent.resolve()
    bundle = Path(sys._MEIPASS)

    # ── 1. Create data directories ──────────────────────────────
    for subdir in ["config/prompts", "config/backups", "logs"]:
        (exe_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ── 2. Prepopulate prompts from bundle if empty ─────────────
    prompts_dir = exe_dir / "config" / "prompts"
    bundle_prompts = bundle / "config" / "prompts"
    if bundle_prompts.exists() and not any(prompts_dir.iterdir()):
        import shutil
        for f in bundle_prompts.iterdir():
            if f.is_file():
                shutil.copy2(f, prompts_dir / f.name)

    # ── 3. Copy config.example.yaml → config.yaml if missing ────
    config_path = exe_dir / "config.yaml"
    if not config_path.exists():
        example = bundle / "config.example.yaml"
        if example.exists():
            import shutil
            shutil.copy2(example, config_path)
            print("=" * 56)
            print("  GrassVision 首次启动")
            print("=" * 56)
            print(f"  已创建默认配置文件:")
            print(f"    {config_path}")
            print()
            print(f"  请编辑该文件，填入你的 API Key 等信息，")
            print(f"  然后重新运行 GrassVision。")
            print("=" * 56)
            sys.exit(0)


def main():
    _ensure_first_run()

    import uvicorn
    from app.config import get_config

    cfg = get_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
