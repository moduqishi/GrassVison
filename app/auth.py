"""Admin authentication via session cookies (no database)."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_config

# In-memory session store: token → expiry_timestamp
_sessions: dict[str, float] = {}
SESSION_DURATION = 3600 * 8  # 8 hours

COOKIE_NAME = "grassvision_session"


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_login(username: str, password: str) -> str | None:
    """Validate credentials.  Returns session token on success, None on failure."""
    cfg = get_config()
    admin_cfg = cfg.admin
    if not admin_cfg.enabled:
        return "admin_disabled"
    if username == admin_cfg.username and _hash_password(password) == _hash_password(admin_cfg.password):
        token = secrets.token_hex(32)
        _sessions[token] = time.time() + SESSION_DURATION
        _clean_expired()
        return token
    return None


def validate_session(token: str) -> bool:
    """Check if a session token is valid."""
    if not token or token == "admin_disabled":
        return token == "admin_disabled"
    expiry = _sessions.get(token)
    if expiry and time.time() < expiry:
        return True
    _sessions.pop(token, None)
    return False


def logout(token: str) -> None:
    _sessions.pop(token, None)


def _clean_expired():
    now = time.time()
    expired = [t for t, exp in _sessions.items() if exp < now]
    for t in expired:
        _sessions.pop(t, None)


class AdminAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that requires login for /admin/* and /api/admin/* routes."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip non-admin paths and the login page itself
        if not (path.startswith("/admin") or path.startswith("/api/admin")):
            return await call_next(request)

        # Allow login page and login API
        if path in ("/admin/login", "/api/admin/login", "/admin/static", "/static"):
            return await call_next(request)

        # Allow static files under /admin
        if path.startswith("/admin/static/"):
            return await call_next(request)

        # Check if admin is disabled
        cfg = get_config()
        if not cfg.admin.enabled:
            return await call_next(request)

        token = request.cookies.get(COOKIE_NAME)
        if not token or not validate_session(token):
            # For API calls, return 401
            if path.startswith("/api/"):
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})
            # For page requests, redirect to login
            return RedirectResponse(url="/admin/login", status_code=302)

        return await call_next(request)
