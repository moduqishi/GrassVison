"""Image extraction, validation, preprocessing, and injection."""
from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

from app.config import get_config
from app.errors import ImageError

# 允许 image/svg+xml、image/heic 等带扩展名的 MIME（\w+ 匹配不了 +）
DATA_URL_RE = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", re.IGNORECASE)

IMG_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",
    b"<svg": "svg+xml",
    b"BM": "bmp",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
}

PREPROCESS_VERSION = "jpeg-rgb-2048-q90-v1"
PREPROCESS_MAX_EDGE = 2048
PREPROCESS_JPEG_QUALITY = 90


# ── Data structures ──────────────────────────────────────────────

@dataclass(frozen=True)
class ImagePosition:
    """Unique position of an image in the messages list."""
    message_index: int
    content_index: int


@dataclass
class ExtractedImage:
    """An image extracted from a message, carrying its position and metadata."""
    position: ImagePosition
    url: str
    is_current: bool  # whether this image is in the last user message

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        if not isinstance(other, ExtractedImage):
            return False
        return self.position == other.position


# ── Extraction ───────────────────────────────────────────────────

def extract_images_from_messages(messages: list) -> list[dict]:
    """Extract flat list of {'url', 'detail'} from all messages (legacy compat)."""
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


def extract_all_images_with_positions(messages: list[dict]) -> list[ExtractedImage]:
    """Extract all images with their (message_index, content_index) positions."""
    result = []
    # Find the last user message index
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for ci, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                img = part.get("image_url", {})
                url = img.get("url", "")
                if url:
                    pos = ImagePosition(message_index=mi, content_index=ci)
                    result.append(ExtractedImage(
                        position=pos,
                        url=url,
                        is_current=(mi == last_user_idx),
                    ))
    return result


def extract_images_from_last_user_message(messages: list[dict]) -> set[ImagePosition]:
    """Return positions of images in the last user message only."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            content = messages[i].get("content")
            positions = set()
            if isinstance(content, list):
                for ci, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        img = part.get("image_url", {})
                        if img.get("url", ""):
                            positions.add(ImagePosition(message_index=i, content_index=ci))
            return positions
    return set()


def extract_current_turn_positions(messages: list[dict]) -> set[ImagePosition]:
    """返回「当前轮次」的图片位置。

    当前轮次 = 最后一次 assistant 消息之后的全部消息（含多条连续用户消息，
    以及 role=tool 的工具返回消息——浏览器工具返回截图等 agent 场景）。
    兼容客户端把「图片」和「文字」分成两条用户消息发送的情况；
    最后一次 assistant 回复之前的图片视为历史，不再处理。
    """
    positions = set()
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            break
        if messages[i].get("role") not in ("user", "tool"):
            continue
        content = messages[i].get("content")
        if not isinstance(content, list):
            continue
        for ci, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                img = part.get("image_url", {})
                if img.get("url", ""):
                    positions.add(ImagePosition(message_index=i, content_index=ci))
    return positions


def extract_user_question(messages: list) -> str:
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


# ── Injection ────────────────────────────────────────────────────

def inject_image_descriptions(
    messages: list[dict],
    replacements: dict[ImagePosition, str],
) -> list[dict]:
    """
    Replace image_url blocks with text descriptions at their exact positions.
    Returns a deep copy of messages with all image_url parts removed and
    text parts injected at the corresponding (message_index, content_index).

    Each original image_url block is replaced by a text block containing its description.
    """
    result = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        new_msg = {**msg}
        if isinstance(content, list):
            new_content = []
            for ci, part in enumerate(content):
                pos = ImagePosition(mi, ci)
                if isinstance(part, dict) and part.get("type") == "image_url":
                    desc = replacements.get(pos)
                    if desc:
                        new_content.append({"type": "text", "text": desc})
                    # If no description, drop the image entirely
                else:
                    new_content.append(part)
            new_msg["content"] = new_content
        # If content is a string, leave it alone
        result.append(new_msg)
    return result


def remove_image_content(messages: list) -> list:
    """Clone messages with image_url parts removed (legacy compat)."""
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


def assert_no_image_url_blocks(messages: list[dict]) -> None:
    """Raise ImageError if any image_url block remains in messages."""
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for ci, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    raise ImageError(
                        f"Unexpected image_url block at message[{mi}].content[{ci}]"
                    )


# ── Preprocessing ────────────────────────────────────────────────

def preprocess_image(raw_bytes: bytes) -> bytes:
    """
    Normalize image for deterministic hashing:
      - EXIF transpose
      - RGBA → white background → RGB
      - Scale long edge to max 2048px
      - Encode as JPEG quality=90, no metadata
    Returns deterministic JPEG bytes.
    """
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)

    # Convert to RGB (handle RGBA / P modes)
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    # Scale if needed
    w, h = img.size
    max_edge = max(w, h)
    if max_edge > PREPROCESS_MAX_EDGE:
        ratio = PREPROCESS_MAX_EDGE / max_edge
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    # Encode to JPEG with deterministic settings
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=PREPROCESS_JPEG_QUALITY,
             optimize=False, progressive=False)
    return buf.getvalue()


def compute_content_hash(image_bytes: bytes) -> str:
    """SHA-256 of preprocessed image bytes."""
    processed = preprocess_image(image_bytes)
    return hashlib.sha256(processed).hexdigest()


# ── Download / Decode / Resolve ──────────────────────────────────

async def fetch_image_bytes(url: str, client: httpx.AsyncClient | None = None) -> bytes:
    cfg = get_config().image
    own_client = client is None
    if own_client:
        # 池化通用下载客户端（连接复用，由 providers 池统一管理）
        from app.providers import get_download_client
        client = get_download_client(cfg.download_timeout)

    try:
        if not cfg.allow_private_network:
            parsed = urlparse(url)
            if parsed.hostname:
                try:
                    addr = ipaddress.ip_address(parsed.hostname)
                    if addr.is_private or addr.is_loopback or addr.is_link_local:
                        raise ImageError(f"Image URL resolves to private network: {parsed.hostname}")
                except ValueError:
                    pass

        resp = await client.get(url, follow_redirects=True)
        if resp.status_code != 200:
            raise ImageError(f"Failed to download image: HTTP {resp.status_code} from {url}")
        data = resp.content
        validate_image_bytes(data, url)
        return data
    except httpx.TimeoutException:
        raise ImageError(f"Image download timed out: {url}")


def decode_base64_image(data_url: str) -> tuple[bytes, str]:
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
    cfg = get_config().image
    max_bytes = cfg.max_image_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise ImageError(f"Image too large: {len(data)/(1024*1024):.1f}MB > {cfg.max_image_size_mb}MB (source: {source})")
    if len(data) == 0:
        raise ImageError(f"Empty image data (source: {source})")
    detected = None
    for sig, fmt in IMG_SIGNATURES.items():
        if data[:len(sig)] == sig:
            detected = fmt
            break
    if detected is None:
        raise ImageError(f"Unrecognized image format (source: {source})")


def get_image_dimensions(data: bytes) -> tuple[int, int]:
    try:
        img = Image.open(io.BytesIO(data))
        return img.size
    except Exception:
        return 0, 0


def validate_image_dimensions(data: bytes) -> None:
    cfg = get_config().image
    w, h = get_image_dimensions(data)
    if w > cfg.max_width or h > cfg.max_height:
        raise ImageError(f"Image dimensions ({w}x{h}) exceed limit ({cfg.max_width}x{cfg.max_height})")


def downscale_image_if_needed(data: bytes) -> tuple[bytes, str]:
    """尺寸超限时降采样（LANCZOS），返回 (新字节, mime)。

    - 不超限或无法解码（SVG 等）时原样返回，mime 为空串表示未变化。
    - 替代"直接拒绝"：与原生多模态一致，大图降采样后仍可分析。
    """
    cfg = get_config().image
    w, h = get_image_dimensions(data)
    if w == 0 or h == 0:
        return data, ""
    if w <= cfg.max_width and h <= cfg.max_height:
        return data, ""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        ratio = min(cfg.max_width / img.width, cfg.max_height / img.height, 1.0)
        if ratio >= 1.0:
            return data, ""
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, ""


async def resolve_image_to_base64(url: str, client: httpx.AsyncClient | None = None) -> str:
    if url.startswith("data:"):
        data, mime = decode_base64_image(url)
        data, new_mime = downscale_image_if_needed(data)
        if new_mime:
            # 已降采样：重新编码为 JPEG data URL
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{new_mime};base64,{b64}"
        return url
    data = await fetch_image_bytes(url, client)
    data, new_mime = downscale_image_if_needed(data)
    mime = "image/png"
    for sig, fmt in IMG_SIGNATURES.items():
        if data[:len(sig)] == sig:
            mime = f"image/{fmt}"
            if fmt == "svg+xml":
                mime = "image/svg+xml"
            break
    if new_mime:
        mime = new_mime
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
