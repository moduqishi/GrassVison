"""Image extraction, validation, and processing."""
from __future__ import annotations

import base64
import io
import ipaddress
import re
import struct
from urllib.parse import urlparse

import httpx
from PIL import Image

from app.config import get_config
from app.errors import ImageError

# Parse Data URL: data:image/png;base64,xxxx
DATA_URL_RE = re.compile(r"^data:(image/\w+);base64,(.+)$", re.IGNORECASE)

# Head bytes for common image formats
IMG_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",  # needs further check
    b"<svg": "svg+xml",
}


def extract_images_from_messages(messages: list) -> list[dict]:
    """
    Extract image entries from OpenAI-format messages.
    Returns list of {'url': str, 'detail': str}
    """
    images = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    img = part.get("image_url", {})
                    url = img.get("url", "")
                    if url:
                        images.append({"url": url, "detail": img.get("detail", "auto")})
    return images


def extract_user_question(messages: list) -> str:
    """Extract the last user text message as the question context for vision."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                if texts:
                    return " ".join(texts)
    return "请描述图片内容"


def remove_image_content(messages: list) -> list:
    """Clone messages with image_url parts removed (for vision-disabled or skip mode)."""
    cleaned = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = [
                p for p in content
                if not (isinstance(p, dict) and p.get("type") == "image_url")
            ]
            cleaned.append({**msg, "content": new_content})
        else:
            cleaned.append(dict(msg))
    return cleaned


async def fetch_image_bytes(url: str, client: httpx.AsyncClient | None = None) -> bytes:
    """Download an image from URL and return bytes.  Validates size and network safety."""
    cfg = get_config().image
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=cfg.download_timeout)

    try:
        # Validate URL is not private network IP
        if not cfg.allow_private_network:
            parsed = urlparse(url)
            if parsed.hostname:
                try:
                    addr = ipaddress.ip_address(parsed.hostname)
                    if addr.is_private or addr.is_loopback or addr.is_link_local:
                        raise ImageError(f"Image URL resolves to private network: {parsed.hostname}")
                except ValueError:
                    # Hostname is a domain name; DNS resolution check deferred to httpx
                    pass

        resp = await client.get(url, follow_redirects=True)
        if resp.status_code != 200:
            raise ImageError(f"Failed to download image: HTTP {resp.status_code} from {url}")

        data = resp.content
        validate_image_bytes(data, url)
        return data
    except httpx.TimeoutException:
        raise ImageError(f"Image download timed out: {url}")
    finally:
        if own_client:
            await client.aclose()


def decode_base64_image(data_url: str) -> tuple[bytes, str]:
    """Decode a data:image/...;base64,... URL into (bytes, mime_type)."""
    m = DATA_URL_RE.match(data_url)
    if not m:
        raise ImageError("Invalid data URL format")
    mime_type = m.group(1)
    try:
        raw = base64.b64decode(m.group(2))
    except Exception:
        raise ImageError("Failed to decode base64 image data")
    validate_image_bytes(raw, "base64 data")
    return raw, mime_type


def validate_image_bytes(data: bytes, source: str = "") -> None:
    """Validate image size and detect format."""
    cfg = get_config().image
    max_bytes = cfg.max_image_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise ImageError(
            f"Image too large: {len(data) / (1024*1024):.1f}MB > {cfg.max_image_size_mb}MB (source: {source})"
        )
    if len(data) == 0:
        raise ImageError(f"Empty image data (source: {source})")

    # Detect format from header bytes
    detected = None
    for sig, fmt in IMG_SIGNATURES.items():
        if data[: len(sig)] == sig:
            detected = fmt
            break
    if detected is None:
        raise ImageError(f"Unrecognized image format (source: {source})")


def get_image_dimensions(data: bytes) -> tuple[int, int]:
    """Get width and height of image bytes. Returns (0,0) on failure."""
    try:
        img = Image.open(io.BytesIO(data))
        return img.size  # (width, height)
    except Exception:
        return 0, 0


def validate_image_dimensions(data: bytes) -> None:
    """Check image dimensions against max_width / max_height limits."""
    cfg = get_config().image
    w, h = get_image_dimensions(data)
    if w > cfg.max_width or h > cfg.max_height:
        raise ImageError(f"Image dimensions ({w}x{h}) exceed limit ({cfg.max_width}x{cfg.max_height})")


async def resolve_image_to_base64(url: str, client: httpx.AsyncClient | None = None) -> str:
    """Resolve an image URL (http/https or data URL) to a base64 data URL string."""
    if url.startswith("data:"):
        # Already base64; validate and return
        data, mime = decode_base64_image(url)
        validate_image_dimensions(data)
        return url

    # HTTP(S) URL
    data = await fetch_image_bytes(url, client)
    validate_image_dimensions(data)

    # Detect MIME
    mime = "image/png"
    for sig, fmt in IMG_SIGNATURES.items():
        if data[: len(sig)] == sig:
            mime = f"image/{fmt}"
            if fmt == "svg+xml":
                mime = "image/svg+xml"
            break

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
