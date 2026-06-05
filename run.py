"""GrassVision standalone entry point for PyInstaller bundles and direct use."""
import sys
from pathlib import Path

# When frozen by PyInstaller, chdir to the bundle root so relative data paths work
if getattr(sys, "frozen", False):
    _mei = Path(sys._MEIPASS)
else:
    _mei = Path(__file__).parent

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
