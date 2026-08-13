"""Tests for the proxy module routing and model resolution."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.proxy import (
    _find_model, _build_source_body, _inject_thinking_guidance, _THINKING_GUIDANCE_TEXT,
    _build_vision_frame,
)
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


class TestThinkingGuidance:
    def test_appends_to_existing_system_message(self):
        msgs = [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "hi"}]
        out = _inject_thinking_guidance(msgs)
        assert len(out) == 2
        assert out[0]["content"].startswith("你是助手")
        assert _THINKING_GUIDANCE_TEXT in out[0]["content"]
        assert out[1]["content"] == "hi"

    def test_prepends_new_system_message_when_none_exists(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = _inject_thinking_guidance(msgs)
        assert len(out) == 2
        assert out[0]["role"] == "system"
        assert out[0]["content"] == _THINKING_GUIDANCE_TEXT
        assert out[1] == msgs[0]

    def test_replaces_non_string_system_content(self):
        msgs = [{"role": "system", "content": [{"type": "text", "text": "x"}]}, {"role": "user", "content": "hi"}]
        out = _inject_thinking_guidance(msgs)
        assert out[0]["role"] == "system"
        assert out[0]["content"] == _THINKING_GUIDANCE_TEXT
        assert len(out) == 2


class TestVisionFrame:
    def test_first_frame_has_role_and_reasoning_content(self):
        frame, is_first = _build_vision_frame("分析中", "deepseek-v4-flash-vision", "cmpl-1", True)
        assert is_first is False
        assert frame.startswith("data: ")
        data = json.loads(frame[6:].strip())
        assert data["model"] == "deepseek-v4-flash-vision"
        delta = data["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        assert delta["reasoning_content"] == "分析中"

    def test_second_frame_has_no_role(self):
        frame, is_first = _build_vision_frame("继续", "m", "cmpl-2", False)
        assert is_first is False
        data = json.loads(frame[6:].strip())
        assert "role" not in data["choices"][0]["delta"]
        assert data["choices"][0]["delta"]["reasoning_content"] == "继续"
