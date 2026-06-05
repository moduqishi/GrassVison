"""GrassVision entry point — used directly or via PyInstaller."""
from __future__ import annotations

import uvicorn
from app.config import get_config


def main():
    cfg = get_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
