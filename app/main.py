"""GrassVision — FastAPI application entry point."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.auth import AdminAuthMiddleware
from app.config import get_config, reload_config, get_config_meta, BASE_DIR
from app.errors import (
    GrassVisionError, ConfigError, ModelNotFoundError, ProviderError,
    ImageError, VisionAnalysisError, grassvision_exception_handler,
)
from app.proxy import handle_chat_completion
from app.schemas import ChatCompletionRequest, ModelInfo
from app.stats import get_stats, flush_stats
from app.admin import router as admin_router


def setup_logging():
    cfg = get_config().logging
    logger = logging.getLogger("grassvision")
    logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if cfg.save_to_file:
        log_file = Path(cfg.file)
        if not log_file.is_absolute():
            log_file = BASE_DIR / log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        reload_config()
        setup_logging()
        logging.getLogger("grassvision").info("GrassVision started")
    except Exception as e:
        print(f"FATAL: Failed to load config: {e}", file=sys.stderr)
    yield
    logging.getLogger("grassvision").info("GrassVision shutting down")
    flush_stats()


app = FastAPI(title="GrassVision", version="1.0.0", lifespan=lifespan)

app.add_middleware(AdminAuthMiddleware)

app.add_exception_handler(GrassVisionError, grassvision_exception_handler)
app.add_exception_handler(ConfigError, grassvision_exception_handler)
app.add_exception_handler(ModelNotFoundError, grassvision_exception_handler)
app.add_exception_handler(ProviderError, grassvision_exception_handler)
app.add_exception_handler(ImageError, grassvision_exception_handler)
app.add_exception_handler(VisionAnalysisError, grassvision_exception_handler)

app.include_router(admin_router)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    logger = logging.getLogger("grassvision")
    start = time.time()

    cfg = get_config()
    if cfg.server.access_key:
        auth_header = raw_request.headers.get("Authorization", "")
        key = auth_header.replace("Bearer ", "")
        if key != cfg.server.access_key:
            raise HTTPException(401, "Invalid access key")

    try:
        url_images = sum(
            1 for m in request.messages
            if isinstance(m.content, list) and
            any(isinstance(p, dict) and p.get("type") == "image_url" for p in m.content)
        ) if request.messages else 0

        response = await handle_chat_completion(request, raw_request)
        elapsed = time.time() - start
        logger.info(
            f"model={request.model} images={url_images} "
            f"stream={request.stream} elapsed={elapsed:.2f}s"
        )
        return response
    except GrassVisionError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(500, f"Internal error: {str(e)}")


@app.get("/v1/models")
async def list_models():
    cfg = get_config()
    models = [
        ModelInfo(id=model_id, owned_by=model_config.name or "grassvision")
        for model_id, model_config in cfg.models.items()
        if model_config.enabled
    ]
    return {"object": "list", "data": [m.model_dump() for m in models]}


@app.get("/health")
async def health():
    cfg = get_config()
    meta = get_config_meta()
    return {
        "status": "ok",
        "version": "1.0.0",
        "config_loaded_at": str(meta.get("loaded_at", "")),
        "config_error": meta.get("error"),
        "models": sum(1 for m in cfg.models.values() if m.enabled),
    }


STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
