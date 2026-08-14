"""Tests for image extraction and validation."""
import base64
import io
import pytest
from PIL import Image
from app.image_utils import (
    extract_images_from_messages, extract_user_question,
    remove_image_content, decode_base64_image, validate_image_bytes,
)
from app.errors import ImageError

TINY_PNG = ("data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class TestImageExtraction:
    def test_extracts_url_image(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]
        }]
        images = extract_images_from_messages(messages)
        assert len(images) == 1
        assert images[0]["url"] == "https://example.com/img.png"
        assert images[0]["detail"] == "auto"

    def test_extracts_base64_image(self):
        messages = [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        }]
        images = extract_images_from_messages(messages)
        assert len(images) == 1
        assert images[0]["url"] == "data:image/png;base64,abc"

    def test_no_images_in_text_only(self):
        messages = [{"role": "user", "content": "just text"}]
        images = extract_images_from_messages(messages)
        assert len(images) == 0

    def test_multiple_images(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,b"}},
            ]
        }]
        images = extract_images_from_messages(messages)
        assert len(images) == 2


class TestUserQuestion:
    def test_extracts_last_user_text(self):
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "help me with this error"},
        ]
        q = extract_user_question(messages)
        assert q == "help me with this error"

    def test_extracts_from_content_array(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what"},
                {"type": "image_url", "image_url": {"url": "data:x"}},
                {"type": "text", "text": "is this"},
            ]
        }]
        q = extract_user_question(messages)
        assert q == "what is this"


class TestRemoveImages:
    def test_removes_image_content(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "https://x.com/img.png"}}
            ]
        }]
        cleaned = remove_image_content(messages)
        assert len(cleaned[0]["content"]) == 1
        assert cleaned[0]["content"][0]["type"] == "text"

    def test_preserves_text_only(self):
        messages = [{"role": "user", "content": "hello"}]
        cleaned = remove_image_content(messages)
        assert cleaned[0]["content"] == "hello"


class TestDecodeBase64:
    def test_valid_data_url(self):
        import base64
        data = base64.b64encode(b"fakeimg").decode()
        url = f"data:image/png;base64,{data}"
        # This should fail validation since "fakeimg" isn't a real image
        with pytest.raises(ImageError):
            decode_base64_image(url)

    def test_invalid_data_url(self):
        with pytest.raises(ImageError):
            decode_base64_image("not a data url")


class TestFormats:
    """P1-11：data URL 正则支持带扩展名的 MIME；BMP/TIFF 签名识别。"""

    def test_data_url_regex_matches_svg_xml(self):
        from app.image_utils import DATA_URL_RE
        m = DATA_URL_RE.match("data:image/svg+xml;base64,PHN2Zz4=")
        assert m is not None
        assert m.group(1) == "image/svg+xml"
        assert m.group(2) == "PHN2Zz4="

    def test_data_url_regex_matches_heic(self):
        from app.image_utils import DATA_URL_RE
        m = DATA_URL_RE.match("data:image/heic;base64,AAAA")
        assert m is not None
        assert m.group(1) == "image/heic"

    def test_bmp_and_tiff_signatures_detected(self):
        from app.image_utils import validate_image_bytes
        validate_image_bytes(b"BM\x00\x00\x00\x00\x00\x00\x00\x00", "bmp-test")
        validate_image_bytes(b"II*\x00\x10\x00\x00\x00", "tiff-test")
        validate_image_bytes(b"MM\x00*\x00\x00\x00\x10", "tiff-test")

    def test_svg_data_url_signature_still_validated(self):
        from app.image_utils import decode_base64_image
        import base64
        svg = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
        raw, mime = decode_base64_image("data:image/svg+xml;base64," + base64.b64encode(svg).decode())
        assert mime == "image/svg+xml"
        assert raw == svg


class TestToolRoleExtraction:
    """P1-7：role=tool 消息中的图片属于当前轮次。"""

    def test_tool_message_image_is_current(self):
        from app.image_utils import extract_current_turn_positions
        from app.image_utils import ImagePosition
        messages = [
            {"role": "user", "content": "打开浏览器"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "b", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]},
        ]
        positions = extract_current_turn_positions(messages)
        assert ImagePosition(message_index=2, content_index=0) in positions

    def test_tool_message_before_assistant_is_historical(self):
        from app.image_utils import extract_current_turn_positions
        messages = [
            {"role": "user", "content": "看这张图"},
            {"role": "tool", "tool_call_id": "c1",
             "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]},
            {"role": "assistant", "content": "看到了"},
            {"role": "user", "content": "继续"},
        ]
        positions = extract_current_turn_positions(messages)
        assert positions == set()


class TestDownscale:
    """超限图片自动降采样，替代直接拒绝。"""

    def test_oversized_image_downscaled(self):
        from app.image_utils import resolve_image_to_base64, get_image_dimensions
        from app.config import get_config
        import asyncio
        cfg = get_config()
        old_w, old_h = cfg.image.max_width, cfg.image.max_height
        cfg.image.max_width, cfg.image.max_height = 32, 32
        try:
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (0, 128, 255)).save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            out = asyncio.run(resolve_image_to_base64(data_url))
            raw, mime = decode_base64_image(out)
            w, h = get_image_dimensions(raw)
            assert w <= 32 and h <= 32, f"应降采样到 32px 内，实际 {w}x{h}"
            assert mime == "image/jpeg"
        finally:
            cfg.image.max_width, cfg.image.max_height = old_w, old_h

    def test_normal_image_unchanged(self):
        from app.image_utils import resolve_image_to_base64
        import asyncio
        out = asyncio.run(resolve_image_to_base64(TINY_PNG))
        assert out == TINY_PNG, "未超限图片应原样返回"
