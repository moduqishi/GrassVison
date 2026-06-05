"""Admin API routes for managing providers, models, prompts, settings, and logs."""
from __future__ import annotations

import base64
from pathlib import Path

from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.auth import check_login, COOKIE_NAME, SESSION_DURATION, validate_session, logout
from app.config import (
    get_config, reload_config, save_config, backup_config,
    read_prompt, write_prompt, delete_prompt, list_prompts,
    get_config_meta, BASE_DIR, BUNDLE_DIR, CONFIG_PATH, PROMPTS_DIR,
)
from app.providers import test_source_connection, test_vision_connection, get_source_client, get_vision_client
from app.schemas import (
    SourceProviderConfig, VisionProviderConfig, EnhancedModelConfig,
    ServerConfig, AdminConfig, ImageConfig, LoggingConfig,
)
from app.vision import analyze_images
from app.stats import get_stats

router = APIRouter()

TEMPLATES_DIR = BUNDLE_DIR / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
    cache_size=0,
)
templates = Jinja2Templates(env=_env)


# ── Cache reconfig helper ─────────────────────────────────────

def _reconfigure_cache_if_changed(old_cache: dict, new_cache_config) -> None:
    """Call ImageCache.reconfigure() only when cache config fields changed."""
    import asyncio
    from app.image_cache import get_image_cache
    try:
        cache = get_image_cache()
        new_enabled = new_cache_config.enabled
        new_ttl = new_cache_config.ttl_seconds
        new_max = new_cache_config.max_entries
        changed = (
            old_cache.get("enabled") != new_enabled or
            old_cache.get("ttl_seconds") != new_ttl or
            old_cache.get("max_entries") != new_max
        )
        if changed:
            asyncio.create_task(cache.reconfigure(
                enabled=new_enabled,
                ttl_seconds=new_ttl,
                max_entries=new_max,
            ))
    except Exception:
        pass  # best-effort, don't break config reload

def _render(request: Request, name: str, context: dict | None = None) -> HTMLResponse:
    ctx = {"request": request}
    if context:
        ctx.update(context)
    return templates.TemplateResponse(request=request, name=name, context=ctx)


def _is_logged_in(request: Request) -> bool:
    cfg = get_config()
    if not cfg.admin.enabled:
        return True
    token = request.cookies.get(COOKIE_NAME)
    return bool(token and validate_session(token))


def _admin_context(request: Request) -> dict:
    cfg = get_config()
    return {
        "config": cfg,
        "meta": get_config_meta(),
        "source_count": len(cfg.source_providers),
        "vision_count": len(cfg.vision_providers),
        "model_count": len(cfg.models),
        "enabled_model_count": sum(1 for m in cfg.models.values() if m.enabled),
        "logged_in": _is_logged_in(request),
    }


# ── Login ────────────────────────────────────────────────────────

@router.get("/admin/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return _render(request, "login.html")


@router.post("/api/admin/login")
async def login_api(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")
    token = check_login(username, password)
    if token:
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE_NAME, token, max_age=SESSION_DURATION, httponly=True)
        return resp
    return JSONResponse({"ok": False, "error": "Invalid credentials"}, status_code=401)


@router.post("/api/admin/logout")
async def logout_api(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        logout(token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME)
    return resp


# ── Dashboard ────────────────────────────────────────────────────

@router.get("/admin", response_class=HTMLResponse)
@router.get("/admin/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return _render(request, "dashboard.html", _admin_context(request))


# ── Source Providers ─────────────────────────────────────────────

@router.get("/admin/source-providers", response_class=HTMLResponse)
async def source_providers_page(request: Request):
    return _render(request, "source_providers.html", _admin_context(request))


@router.get("/api/admin/source-providers")
async def list_source_providers():
    cfg = get_config()
    return {"providers": {k: v.model_dump() for k, v in cfg.source_providers.items()}}


@router.post("/api/admin/source-providers")
async def create_source_provider(data: dict):
    cfg = get_config()
    key = data.get("key", "")
    if not key:
        raise HTTPException(400, "Provider key is required")
    if key in cfg.source_providers:
        raise HTTPException(400, f"Provider '{key}' already exists")
    cfg.source_providers[key] = SourceProviderConfig(**data)
    save_config(cfg)
    reload_config()
    return {"ok": True}


@router.put("/api/admin/source-providers/{key}")
async def update_source_provider(key: str, data: dict):
    cfg = get_config()
    if key not in cfg.source_providers:
        raise HTTPException(404, f"Provider '{key}' not found")
    if data.get("api_key", "") == "":
        data["api_key"] = cfg.source_providers[key].api_key
    cfg.source_providers[key] = SourceProviderConfig(**data)
    save_config(cfg)
    reload_config()
    return {"ok": True}


@router.delete("/api/admin/source-providers/{key}")
async def delete_source_provider(key: str):
    cfg = get_config()
    if key not in cfg.source_providers:
        raise HTTPException(404, f"Provider '{key}' not found")
    dependents = [mid for mid, m in cfg.models.items() if m.source_provider == key]
    if dependents:
        raise HTTPException(400, f"Cannot delete: used by models: {', '.join(dependents)}")
    del cfg.source_providers[key]
    save_config(cfg)
    reload_config()
    return {"ok": True}


@router.post("/api/admin/source-providers/{key}/test")
async def test_source_provider(key: str):
    cfg = get_config()
    if key not in cfg.source_providers:
        raise HTTPException(404)
    result = await test_source_connection(cfg.source_providers[key])
    return {"ok": result["ok"], "result": result}


@router.get("/api/admin/source-providers/{key}/models")
async def fetch_source_models(key: str):
    """Fetch available model IDs from a source provider's /models endpoint."""
    cfg = get_config()
    if key not in cfg.source_providers:
        raise HTTPException(404)
    provider = cfg.source_providers[key]
    client = get_source_client(provider)
    try:
        resp = await client.get("/models", timeout=15)
        if resp.status_code != 200:
            return {"ok": True, "models": [], "error": f"HTTP {resp.status_code}"}
        data = resp.json()
        model_ids = [m["id"] for m in data.get("data", [])]
        return {"ok": True, "models": model_ids}
    except Exception as e:
        return {"ok": True, "models": [], "error": str(e)}
    finally:
        await client.aclose()


# ── Vision Providers ─────────────────────────────────────────────

@router.get("/admin/vision-providers", response_class=HTMLResponse)
async def vision_providers_page(request: Request):
    return _render(request, "vision_providers.html", _admin_context(request))


@router.get("/api/admin/vision-providers")
async def list_vision_providers():
    cfg = get_config()
    return {"providers": {k: v.model_dump() for k, v in cfg.vision_providers.items()}}


@router.post("/api/admin/vision-providers")
async def create_vision_provider(data: dict):
    cfg = get_config()
    key = data.get("key", "")
    if not key:
        raise HTTPException(400, "Provider key is required")
    if key in cfg.vision_providers:
        raise HTTPException(400, f"Provider '{key}' already exists")
    cfg.vision_providers[key] = VisionProviderConfig(**data)
    save_config(cfg)
    reload_config()
    return {"ok": True}


@router.put("/api/admin/vision-providers/{key}")
async def update_vision_provider(key: str, data: dict):
    cfg = get_config()
    if key not in cfg.vision_providers:
        raise HTTPException(404)
    if data.get("api_key", "") == "":
        data["api_key"] = cfg.vision_providers[key].api_key
    cfg.vision_providers[key] = VisionProviderConfig(**data)
    save_config(cfg)
    reload_config()
    return {"ok": True}


@router.delete("/api/admin/vision-providers/{key}")
async def delete_vision_provider(key: str):
    cfg = get_config()
    if key not in cfg.vision_providers:
        raise HTTPException(404)
    dependents = [mid for mid, m in cfg.models.items() if m.vision_provider == key]
    if dependents:
        raise HTTPException(400, f"Cannot delete: used by models: {', '.join(dependents)}")
    del cfg.vision_providers[key]
    save_config(cfg)
    reload_config()
    return {"ok": True}


@router.post("/api/admin/vision-providers/{key}/test")
async def test_vision_provider(key: str, file: UploadFile = File(None), question: str = Form("")):
    cfg = get_config()
    if key not in cfg.vision_providers:
        raise HTTPException(404)
    if not file:
        result = await test_vision_connection(cfg.vision_providers[key])
        return {"ok": result["ok"], "type": "connectivity", "result": result}

    img_bytes = await file.read()
    b64 = base64.b64encode(img_bytes).decode("ascii")
    ext = Path(file.filename or "img.png").suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    data_url = f"data:{mime};base64,{b64}"

    provider = cfg.vision_providers[key]
    vision_result = await analyze_images(
        messages=[{"role": "user", "content": question or "请描述这张图片"}],
        image_urls=[data_url],
        vision_provider_key=key,
        vision_model=provider.model,
        vision_prompt="prompts/default.txt",
    )
    return {"ok": True, "type": "vision_test", "result": vision_result}


@router.get("/api/admin/vision-providers/{key}/models")
async def fetch_vision_models(key: str):
    """Fetch available model IDs from a vision provider's /models endpoint."""
    cfg = get_config()
    if key not in cfg.vision_providers:
        raise HTTPException(404)
    provider = cfg.vision_providers[key]
    client = get_vision_client(provider)
    try:
        resp = await client.get("/models", timeout=15)
        if resp.status_code != 200:
            return {"ok": True, "models": [], "error": f"HTTP {resp.status_code}"}
        data = resp.json()
        model_ids = [m["id"] for m in data.get("data", [])]
        return {"ok": True, "models": model_ids}
    except Exception as e:
        return {"ok": True, "models": [], "error": str(e)}
    finally:
        await client.aclose()


# ── Enhanced Models ──────────────────────────────────────────────

@router.get("/admin/models", response_class=HTMLResponse)
async def models_page(request: Request):
    ctx = _admin_context(request)
    ctx["source_keys"] = list(get_config().source_providers.keys())
    ctx["vision_keys"] = list(get_config().vision_providers.keys())
    ctx["prompt_files"] = list_prompts()
    return _render(request, "models.html", ctx)


@router.get("/api/admin/models")
async def list_models():
    cfg = get_config()
    return {"models": {k: v.model_dump() for k, v in cfg.models.items()}}


@router.post("/api/admin/models")
async def create_model(data: dict):
    cfg = get_config()
    model_id = data.get("model_id", "")
    if not model_id:
        raise HTTPException(400, "Model ID is required")
    if model_id in cfg.models:
        raise HTTPException(400, f"Model '{model_id}' already exists")
    data.pop("model_id", None)
    cfg.models[model_id] = EnhancedModelConfig(**data)
    save_config(cfg)
    reload_config()
    return {"ok": True}


@router.put("/api/admin/models/{model_id}")
async def update_model(model_id: str, data: dict):
    cfg = get_config()
    if model_id not in cfg.models:
        raise HTTPException(404)
    data.pop("model_id", None)
    cfg.models[model_id] = EnhancedModelConfig(**data)
    save_config(cfg)
    reload_config()
    return {"ok": True}


@router.delete("/api/admin/models/{model_id}")
async def delete_model(model_id: str):
    cfg = get_config()
    if model_id not in cfg.models:
        raise HTTPException(404)
    del cfg.models[model_id]
    save_config(cfg)
    reload_config()
    return {"ok": True}


# ── Prompts ──────────────────────────────────────────────────────

@router.get("/admin/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request):
    ctx = _admin_context(request)
    ctx["prompts"] = list_prompts()
    return _render(request, "prompts.html", ctx)


@router.get("/api/admin/prompts")
async def api_list_prompts():
    return {"prompts": list_prompts()}


@router.get("/api/admin/prompts/{filename}")
async def api_get_prompt(filename: str):
    content = read_prompt(filename)
    return {"filename": filename, "content": content}


@router.post("/api/admin/prompts")
async def api_create_prompt(data: dict):
    filename = data.get("filename", "")
    content = data.get("content", "")
    if not filename or not filename.endswith(".txt"):
        raise HTTPException(400, "Filename must end with .txt")
    write_prompt(filename, content)
    return {"ok": True}


@router.put("/api/admin/prompts/{filename}")
async def api_update_prompt(filename: str, data: dict):
    content = data.get("content", "")
    write_prompt(filename, content)
    return {"ok": True}


@router.delete("/api/admin/prompts/{filename}")
async def api_delete_prompt(filename: str):
    delete_prompt(filename)
    return {"ok": True}


# ── Playground ──────────────────────────────────────────────────

@router.get("/admin/playground", response_class=HTMLResponse)
async def playground_page(request: Request):
    ctx = _admin_context(request)
    ctx["models"] = list(get_config().models.keys())
    return _render(request, "playground.html", ctx)


# ── Settings ────────────────────────────────────────────────────

@router.get("/admin/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return _render(request, "settings.html", _admin_context(request))


@router.put("/api/admin/settings")
async def update_settings(data: dict):
    cfg = get_config()
    old_cache = cfg.image.vision_cache.model_dump() if cfg else {}
    if "server" in data:
        cfg.server = ServerConfig(**data["server"])
    if "admin" in data:
        ad = data["admin"]
        cfg.admin = AdminConfig(**ad) if ad.get("password") else AdminConfig(**{**ad, "password": cfg.admin.password})
    if "image" in data:
        cfg.image = ImageConfig(**data["image"])
    if "logging" in data:
        cfg.logging = LoggingConfig(**data["logging"])
    save_config(cfg)
    reload_config()
    # Reconfigure cache only if cache config actually changed
    from app.image_cache import get_image_cache as _get_cache
    new_cache = cfg.image.vision_cache
    _reconfigure_cache_if_changed(old_cache, new_cache)
    needs_restart = "server" in data
    return {"ok": True, "needs_restart": needs_restart}


# ── Config Preview ──────────────────────────────────────────────

@router.get("/admin/config-preview", response_class=HTMLResponse)
async def config_preview_page(request: Request):
    ctx = _admin_context(request)
    ctx["config_yaml"] = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    return _render(request, "config_preview.html", ctx)


@router.put("/api/admin/config/yaml")
async def save_yaml_config(data: dict):
    import yaml
    yaml_str = data.get("yaml", "")
    try:
        yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")
    backup_config()
    CONFIG_PATH.write_text(yaml_str, encoding="utf-8")
    reload_config()
    return {"ok": True}


@router.post("/api/admin/reload")
async def api_reload_config():
    cfg = get_config()
    old_cache = cfg.image.vision_cache.model_dump() if cfg else {}
    reload_config()
    from app.image_cache import get_image_cache as _get_cache
    _reconfigure_cache_if_changed(old_cache, get_config().image.vision_cache)
    return {"ok": True}


@router.get("/api/admin/config/download")
async def download_config():
    content = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else "# empty"
    return PlainTextResponse(content, media_type="application/x-yaml",
                             headers={"Content-Disposition": "attachment; filename=config.yaml"})


# ── Logs ─────────────────────────────────────────────────────────

@router.get("/admin/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    return _render(request, "logs.html", _admin_context(request))


@router.get("/api/admin/logs")
async def api_get_logs(lines: int = 200, keyword: str = "", error_only: bool = False):
    cfg = get_config()
    log_file = Path(cfg.logging.file)
    if not log_file.is_absolute():
        log_file = BASE_DIR / log_file
    if not log_file.exists():
        return {"lines": [], "total": 0}
    content = log_file.read_text(encoding="utf-8", errors="replace")
    all_lines = content.strip().split("\n")
    if keyword:
        all_lines = [l for l in all_lines if keyword.lower() in l.lower()]
    if error_only:
        all_lines = [l for l in all_lines if "ERROR" in l or "error" in l]
    total = len(all_lines)
    tail = all_lines[-lines:]
    return {"lines": tail, "total": total}


@router.delete("/api/admin/logs")
async def clear_logs():
    cfg = get_config()
    log_file = Path(cfg.logging.file)
    if not log_file.is_absolute():
        log_file = BASE_DIR / log_file
    log_file.write_text("")
    return {"ok": True}


# ── Stats ────────────────────────────────────────────────────────

@router.get("/admin/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    return _render(request, "stats.html", _admin_context(request))


@router.get("/api/admin/stats")
async def api_stats():
    """Return usage statistics summary, recent call history, and cache stats."""
    s = get_stats()
    from app.image_cache import get_image_cache as _get_cache
    cache_stats = await _get_cache().stats()
    return {
        "summary": s.summary(),
        "recent_calls": s.recent_calls(50),
        "cache": cache_stats,
    }


@router.post("/api/admin/stats/reset")
async def api_reset_stats():
    get_stats().reset()
    from app.image_cache import get_image_cache as _get_cache
    await _get_cache().clear()
    return {"ok": True}
