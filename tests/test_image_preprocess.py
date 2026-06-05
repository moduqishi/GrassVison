"""Tests for image preprocessing and position extraction."""
import struct, zlib
import pytest
from app.image_utils import (
    preprocess_image, compute_content_hash,
    extract_all_images_with_positions,
    extract_images_from_last_user_message,
    inject_image_descriptions,
    assert_no_image_url_blocks,
    ImagePosition,
)
from app.errors import ImageError


def _make_png(width=10, height=10):
    """Create a minimal valid PNG of given size (solid red)."""
    sig = b'\x89PNG\r\n\x1a\n'

    def chunk(ctype, data):
        c = ctype + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    # Raw image data: filter byte 0 + RGB row
    raw = b''
    for _ in range(height):
        raw += b'\x00' + b'\xff\x00\x00' * width

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    compressed = zlib.compress(raw)
    return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')


class TestPreprocess:
    def test_preprocess_returns_deterministic(self):
        png1 = _make_png(10, 10)
        png2 = _make_png(10, 10)
        result1 = preprocess_image(png1)
        result2 = preprocess_image(png2)
        assert result1 == result2  # same content → same output
        assert result1[:2] == b'\xff\xd8'  # JPEG header

    def test_content_hash_deterministic(self):
        png = _make_png(20, 20)
        h1 = compute_content_hash(png)
        h2 = compute_content_hash(png)
        assert h1 == h2

    def test_different_images_different_hash(self):
        png1 = _make_png(10, 10)
        png2 = _make_png(11, 10)
        h1 = compute_content_hash(png1)
        h2 = compute_content_hash(png2)
        assert h1 != h2


class TestImagePositions:
    def test_extract_all_with_positions(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "q1"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
            ]},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,b"}},
                {"type": "text", "text": "q2"},
            ]},
        ]
        imgs = extract_all_images_with_positions(messages)
        assert len(imgs) == 2
        assert imgs[0].position == ImagePosition(0, 1)
        assert imgs[1].position == ImagePosition(2, 0)
        # The last user message is at index 2
        assert imgs[0].is_current is False
        assert imgs[1].is_current is True

    def test_last_user_message_images(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "img1"}},
            ]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [
                {"type": "text", "text": "follow up"},
                {"type": "image_url", "image_url": {"url": "img2"}},
                {"type": "image_url", "image_url": {"url": "img3"}},
            ]},
        ]
        positions = extract_images_from_last_user_message(messages)
        assert len(positions) == 2
        assert ImagePosition(2, 1) in positions
        assert ImagePosition(2, 2) in positions
        assert ImagePosition(0, 0) not in positions

    def test_inject_descriptions(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "img1"}},
            ]},
        ]
        replacements = {ImagePosition(0, 1): "**图片描述**"}
        result = inject_image_descriptions(messages, replacements)
        content = result[0]["content"]
        assert len(content) == 2  # text + injected text
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "hello"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "**图片描述**"

    def test_assert_no_images_passes(self):
        messages = [{"role": "user", "content": "no images here"}]
        assert_no_image_url_blocks(messages)  # should not raise

    def test_assert_no_images_fails(self):
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "img"}},
        ]}]
        with pytest.raises(ImageError):
            assert_no_image_url_blocks(messages)
