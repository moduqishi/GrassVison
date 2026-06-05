"""Tests for the proxy module routing and model resolution."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.proxy import _find_model, _build_source_body
from app.schemas import ChatCompletionRequest, ChatMessage, EnhancedModelConfig
from app.errors import ModelNotFoundError
from app.config import get_config


class TestFindModel:
    def test_finds_existing_model(self):
        model = _find_model("deepseek-vision")
        assert model is not None
        assert model.source_model == "deepseek-chat"

    def test_raises_for_unknown_model(self):
        with pytest.raises(ModelNotFoundError):
            _find_model("nonexistent-model")


class TestBuildBody:
    def test_builds_source_body(self):
        request = ChatCompletionRequest(
            model="deepseek-vision",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.7,
            max_tokens=100,
        )
        messages = [{"role": "user", "content": "hello"}]
        model = EnhancedModelConfig(
            source_model="deepseek-chat",
            source_provider="deepseek",
            replace_response_model=True,
        )
        body = _build_source_body(request, model, messages)
        assert body["model"] == "deepseek-chat"
        assert body["stream"] is False
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 100
        assert body["messages"][0]["content"] == "hello"

    def test_omits_none_params(self):
        request = ChatCompletionRequest(
            model="deepseek-vision",
            messages=[ChatMessage(role="user", content="hi")],
        )
        messages = [{"role": "user", "content": "hi"}]
        model = EnhancedModelConfig(source_model="deepseek-chat")
        body = _build_source_body(request, model, messages)
        assert "temperature" not in body
        assert "top_p" not in body
        assert body["messages"][0]["role"] == "user"
