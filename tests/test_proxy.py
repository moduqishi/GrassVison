"""Tests for the proxy module routing and model resolution."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.proxy import _find_model, _build_source_body
from app.schemas import ChatCompletionRequest, ChatMessage, EnhancedModelConfig
from app.errors import ModelNotFoundError
from app.config import get_config


class TestFindModel:
    def test_finds_existing_model(self):
        from app.config import get_config
        cfg = get_config()
        model_id = next(iter(cfg.models.keys()))
        model = _find_model(model_id)
        assert model is not None
        assert model.source_model == cfg.models[model_id].source_model

    def test_raises_for_unknown_model(self):
        with pytest.raises(ModelNotFoundError):
            _find_model("nonexistent-model")


class TestBuildBody:
    def test_builds_source_body(self):
        from app.config import get_config
        cfg = get_config()
        model_id = list(cfg.models.keys())[0]
        request = ChatCompletionRequest(
            model=model_id,
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
