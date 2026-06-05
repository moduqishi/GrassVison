"""Tests for image extraction and validation."""
import pytest
from app.image_utils import (
    extract_images_from_messages, extract_user_question,
    remove_image_content, decode_base64_image, validate_image_bytes,
)
from app.errors import ImageError


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
