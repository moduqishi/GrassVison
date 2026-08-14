"""Tests for the proxy module routing and model resolution."""
import base64
import io
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
from app.proxy import (
    _find_model, _build_source_body, _inject_thinking_guidance, _THINKING_GUIDANCE_TEXT,
    _build_vision_frame, _vision_usage_extra,
)
from app.schemas import ChatCompletionRequest, ChatMessage, EnhancedModelConfig, VisionProviderConfig
from app.errors import ModelNotFoundError, VisionAnalysisError
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


TINY_PNG = ("data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _make_png_data_url(rgb):
    """用 PIL 生成一张确定性的 2x2 PNG data URL（保证有效可解码）。"""
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


TINY_PNG2 = _make_png_data_url((255, 0, 0))


def _make_big_png_data_url(size=(100, 100), rgb=(10, 20, 30)):
    """生成 size 尺寸的 PNG data URL（grounding 裁剪需要足够大的图像）。"""
    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class _FakeResp:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class _FakeSourceClient:
    def __init__(self):
        self.body = None

    async def post(self, url, json=None):
        self.body = json
        return _FakeResp(200, {"choices": [{"message": {"content": "ok"}}], "usage": {}})

    async def aclose(self):
        pass


class _FakeVisionClient:
    def __init__(self):
        self.post_calls = []

    async def post(self, url, json=None):
        self.post_calls.append(json)
        return _FakeResp(200, {"choices": [{"message": {"content": "## 图片 1\n测试描述"}}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    async def aclose(self):
        pass


class _RawReq:
    pass


class TestCurrentImageOnly:
    """只处理当前（最后一条用户消息）图片：历史图片剥离、不调视觉、不注入上下文。"""

    def test_historical_image_stripped_without_vision(self):
        from app.proxy import handle_chat_completion
        request = ChatCompletionRequest(
            model="openai-vision",
            messages=[
                ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ]),
                ChatMessage(role="assistant", content="这是一张测试图。"),
                ChatMessage(role="user", content="继续解释一下"),
            ],
            stream=False,
        )
        vision = _FakeVisionClient()
        source = _FakeSourceClient()
        with patch("app.vision.get_vision_client", return_value=vision), \
             patch("app.proxy.get_source_client", return_value=source):
            import asyncio
            asyncio.run(handle_chat_completion(request, _RawReq()))
        assert vision.post_calls == []
        joined = json.dumps(source.body["messages"], ensure_ascii=False)
        assert "image_url" not in joined
        assert "grassvision_image_context" not in joined
        assert "测试描述" not in joined

    def test_current_image_analyzed_and_injected(self):
        from app.proxy import handle_chat_completion
        request = ChatCompletionRequest(
            model="openai-vision",
            messages=[ChatMessage(role="user", content=[
                {"type": "image_url", "image_url": {"url": TINY_PNG}},
                {"type": "text", "text": "这张图是什么?"},
            ])],
            stream=False,
        )
        vision = _FakeVisionClient()
        source = _FakeSourceClient()
        with patch("app.vision.get_vision_client", return_value=vision), \
             patch("app.proxy.get_source_client", return_value=source):
            import asyncio
            asyncio.run(handle_chat_completion(request, _RawReq()))
        assert len(vision.post_calls) == 1
        joined = json.dumps(source.body["messages"], ensure_ascii=False)
        assert "image_url" not in joined
        assert "测试描述" in joined
        assert "grassvision_image_context" in joined

    def test_image_and_text_in_separate_messages_still_analyzed(self):
        """客户端把图片和文字分成两条用户消息发（无 assistant 回复间隔）→ 图片仍视为当前。"""
        from app.proxy import handle_chat_completion
        request = ChatCompletionRequest(
            model="openai-vision",
            messages=[
                ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                ]),
                ChatMessage(role="user", content="这张图是什么?"),
            ],
            stream=False,
        )
        vision = _FakeVisionClient()
        source = _FakeSourceClient()
        with patch("app.vision.get_vision_client", return_value=vision), \
             patch("app.proxy.get_source_client", return_value=source):
            import asyncio
            asyncio.run(handle_chat_completion(request, _RawReq()))
        joined = json.dumps(source.body["messages"], ensure_ascii=False)
        assert "测试描述" in joined, "分条发送的图片应被当作当前图片并注入描述"
        assert "grassvision_image_context" in joined

    def test_images_before_last_assistant_reply_are_historical(self):
        """最后一条 assistant 回复之前的图片 → 属于更早轮次，剥离且不调视觉。"""
        from app.proxy import handle_chat_completion
        request = ChatCompletionRequest(
            model="openai-vision",
            messages=[
                ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "看这张图"},
                ]),
                ChatMessage(role="assistant", content="看到了。"),
                ChatMessage(role="user", content="继续"),
            ],
            stream=False,
        )
        vision = _FakeVisionClient()
        source = _FakeSourceClient()
        with patch("app.vision.get_vision_client", return_value=vision), \
             patch("app.proxy.get_source_client", return_value=source):
            import asyncio
            asyncio.run(handle_chat_completion(request, _RawReq()))
        assert vision.post_calls == []
        joined = json.dumps(source.body["messages"], ensure_ascii=False)
        assert "image_url" not in joined
        assert "grassvision_image_context" not in joined


class _FailingVisionClient:
    """视觉调用总是失败的客户端，用于测试 failure_mode。"""

    def __init__(self):
        self.post_calls = []

    async def post(self, url, json=None):
        from app.errors import VisionAnalysisError
        self.post_calls.append(json)
        raise VisionAnalysisError("provider down")

    async def aclose(self):
        pass


def _clear_image_cache():
    """清空全局图片缓存，保证测试隔离。"""
    from app.image_cache import get_image_cache
    import asyncio
    cache = get_image_cache()
    asyncio.run(cache.clear())


class TestVisionFailureMode:
    """vision_failure_mode: error → 502；skip → 剥离图片 + 注入失败说明继续。"""

    def test_error_mode_returns_502(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = model.vision_failure_mode
        model.vision_failure_mode = "error"
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ])],
                stream=False,
            )
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=_FailingVisionClient()), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))
            assert resp.status_code == 502, "error 模式视觉失败应返回 502"
            assert source.body is None, "502 时不应转发源模型"
        finally:
            model.vision_failure_mode = old
            _clear_image_cache()

    def test_skip_mode_injects_failure_note_and_continues(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = model.vision_failure_mode
        model.vision_failure_mode = "skip"
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ])],
                stream=False,
            )
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=_FailingVisionClient()), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))
            assert resp.status_code == 200
            joined = json.dumps(source.body["messages"], ensure_ascii=False)
            assert "image_url" not in joined, "skip 模式必须先剥离 image_url"
            assert "视觉分析失败" in joined, "skip 模式应注入失败说明（fail-open）"
        finally:
            model.vision_failure_mode = old
            _clear_image_cache()


class TestQuestionAwareCache:
    """question_aware_cache=true：视觉模型收到真实用户问题，prompt 占位符被替换。"""

    def test_prompt_contains_real_question(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old = cfg.image.question_aware_cache
        cfg.image.question_aware_cache = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            assert len(vision.post_calls) == 1
            payload = vision.post_calls[0]
            system_prompt = payload["messages"][0]["content"]
            assert "这张图是什么?" in system_prompt, "视觉 prompt 应包含真实用户问题"
            assert "{user_question}" not in system_prompt, "占位符应被替换"
        finally:
            cfg.image.question_aware_cache = old
            _clear_image_cache()


class TestHistoricalCacheReuse:
    """reuse_historical_cache=true：历史图片缓存命中 → 描述原地注入，不触发新分析。"""

    def test_historical_image_uses_cached_description(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old = cfg.image.reuse_historical_cache
        cfg.image.reuse_historical_cache = True
        try:
            # 请求 1：图片在当前轮 → 分析并写入缓存
            request1 = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "看这张图"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request1, _RawReq()))
            assert len(vision.post_calls) == 1

            # 请求 2：同一张图变为历史（assistant 之后是纯文字追问）→ 缓存命中
            request2 = ChatCompletionRequest(
                model="openai-vision",
                messages=[
                    ChatMessage(role="user", content=[
                        {"type": "image_url", "image_url": {"url": TINY_PNG}},
                        {"type": "text", "text": "看这张图"},
                    ]),
                    ChatMessage(role="assistant", content="看到了。"),
                    ChatMessage(role="user", content="继续解释一下"),
                ],
                stream=False,
            )
            vision2 = _FakeVisionClient()
            source2 = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision2), \
                 patch("app.proxy.get_source_client", return_value=source2):
                import asyncio
                asyncio.run(handle_chat_completion(request2, _RawReq()))
            assert vision2.post_calls == [], "历史图片缓存命中不应触发新的视觉调用"
            joined = json.dumps(source2.body["messages"], ensure_ascii=False)
            assert "测试描述" in joined, "历史图片的缓存描述应原地注入"
            assert "image_url" not in joined
        finally:
            cfg.image.reuse_historical_cache = old
            _clear_image_cache()


class TestToolRoleImages:
    """role=tool 消息中的图片（agent 工具返回截图）应视为当前图片并分析。"""

    def test_tool_message_image_is_analyzed(self):
        from app.proxy import handle_chat_completion
        request = ChatCompletionRequest(
            model="openai-vision",
            messages=[
                ChatMessage(role="user", content="请打开浏览器截图"),
                ChatMessage(role="assistant", content=None, tool_calls=[
                    {"id": "c1", "type": "function",
                     "function": {"name": "browse", "arguments": "{}"}},
                ]),
                ChatMessage(role="tool", tool_call_id="c1", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                ]),
            ],
            stream=False,
        )
        vision = _FakeVisionClient()
        source = _FakeSourceClient()
        with patch("app.vision.get_vision_client", return_value=vision), \
             patch("app.proxy.get_source_client", return_value=source):
            import asyncio
            asyncio.run(handle_chat_completion(request, _RawReq()))
        assert len(vision.post_calls) == 1, "tool 消息图片应触发视觉分析"
        joined = json.dumps(source.body["messages"], ensure_ascii=False)
        assert "测试描述" in joined, "tool 图片的描述应注入"
        assert "image_url" not in joined


class TestMultiImageCombined:
    """multi_image_mode=combined：多图合并为一次视觉调用，结果去重注入。"""

    def test_two_images_one_vision_call(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old = cfg.image.multi_image_mode
        cfg.image.multi_image_mode = "combined"
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "image_url", "image_url": {"url": TINY_PNG2}},
                    {"type": "text", "text": "对比这两张图"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            assert len(vision.post_calls) == 1, "联合分析应只有一次视觉调用"
            payload = vision.post_calls[0]
            image_parts = [p for p in payload["messages"][1]["content"]
                           if p.get("type") == "image_url"]
            assert len(image_parts) == 2, "一次调用应同时携带两张图"
            joined = json.dumps(source.body["messages"], ensure_ascii=False)
            assert "测试描述" in joined
            assert "image_url" not in joined
        finally:
            cfg.image.multi_image_mode = old
            _clear_image_cache()

    def test_auto_mode_combines_on_comparison_intent(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        from app.vision import _detect_comparison_intent
        assert _detect_comparison_intent("对比这两张图有什么区别")
        assert _detect_comparison_intent("which one is better?")
        assert not _detect_comparison_intent("描述这张图片")


class _FailoverVisionClient:
    """第一次调用失败、第二次成功，用于测试故障转移。"""

    def __init__(self):
        self.calls = 0

    async def post(self, url, json=None):
        self.calls += 1
        if self.calls == 1:
            raise VisionAnalysisError("primary provider down")
        return _FakeResp(200, {"choices": [{"message": {"content": "failover 描述"}}],
                               "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})

    async def aclose(self):
        pass


class TestVisionFailover:
    """vision_provider_failover：主渠道失败自动切换到备用渠道。"""

    def test_failover_uses_backup_provider(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old_failover = model.vision_provider_failover
        old_providers = dict(cfg.vision_providers)
        model.vision_provider_failover = ["openai-backup"]
        cfg.vision_providers["openai-backup"] = VisionProviderConfig(
            name="openai-backup", enabled=True,
            base_url="https://example.com/v1", api_key="k", model="backup-vl",
        )
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ])],
                stream=False,
            )
            vision = _FailoverVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))
            assert resp.status_code == 200
            assert vision.calls == 2, "主渠道失败后应尝试备用渠道"
            joined = json.dumps(source.body["messages"], ensure_ascii=False)
            assert "failover 描述" in joined, "应注入备用渠道的分析结果"
        finally:
            model.vision_provider_failover = old_failover
            cfg.vision_providers = old_providers
            _clear_image_cache()


class TestUsageTransparency:
    """响应 usage 应透传视觉模型 token（vision_* 前缀，不覆盖标准字段）。"""

    def test_non_stream_usage_contains_vision_tokens(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))
            data = json.loads(resp.body)
            usage = data["usage"]
            assert usage.get("vision_prompt_tokens") == 10
            assert usage.get("vision_total_tokens") == 15
            assert _vision_usage_extra(None) is None
            assert _vision_usage_extra({}) is None
        finally:
            _clear_image_cache()


class _ScriptedVisionClient:
    """按顺序返回预设结果的视觉客户端（用于 grounding/长截图多阶段测试）。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.post_calls = []
        self.calls = 0

    async def post(self, url, json=None):
        self.post_calls.append(json)
        resp_text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return _FakeResp(200, {"choices": [{"message": {"content": resp_text}}], "usage": {}})

    async def aclose(self):
        pass


class TestGroundingZoom:
    """grounding_zoom：先定位坐标框，再裁剪放大二次精读。"""

    def test_two_stage_grounding_and_zoom(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = cfg.image.grounding_zoom
        cfg.image.grounding_zoom = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": _make_big_png_data_url()}},
                    {"type": "text", "text": "这个按钮的颜色是什么?"},
                ])],
                stream=False,
            )
            # 调用 1：单图初步分析；调用 2（定位）：返回坐标框；调用 3（放大精读）：返回细节
            vision = _ScriptedVisionClient([
                "初步描述",
                "x1: 100\ny1: 200\nx2: 300\ny2: 400",
                "放大后的按钮是蓝色的。",
            ])
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            assert vision.calls == 3, "grounding 应触发三次视觉调用（初析 + 定位 + 放大）"
            joined = json.dumps(source.body["messages"], ensure_ascii=False)
            assert "目标元素放大分析" in joined
            assert "蓝色" in joined
            assert "x1=100" in joined
        finally:
            cfg.image.grounding_zoom = old
            _clear_image_cache()

    def test_no_grounding_without_element_intent(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        from app.vision import _has_grounding_intent
        assert _has_grounding_intent("这个按钮在哪")
        assert _has_grounding_intent("click the button")
        assert not _has_grounding_intent("描述整张图的氛围")
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = cfg.image.grounding_zoom
        cfg.image.grounding_zoom = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": _make_big_png_data_url()}},
                    {"type": "text", "text": "描述整张图的氛围"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            assert len(vision.post_calls) == 1, "无元素意图不应触发 grounding 二次调用"
        finally:
            cfg.image.grounding_zoom = old
            _clear_image_cache()


class TestLongScreenshotOCR:
    """long_screenshot_ocr：高宽比≥3 的长截图自动分段分析合并。"""

    def _tall_png(self, height=2400, width=200):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def test_tall_image_sliced_into_bands(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old = cfg.image.long_screenshot_ocr
        cfg.image.long_screenshot_ocr = True
        try:
            tall = self._tall_png(height=2400, width=200)
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": tall}},
                    {"type": "text", "text": "提取这段聊天记录"},
                ])],
                stream=False,
            )
            # 第一次调用：单图分析；之后每段一次调用
            vision = _ScriptedVisionClient(["初始描述", "第1段内容", "第2段内容", "第3段内容"])
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            assert vision.calls >= 3, "长截图应产生多次分段调用"
            joined = json.dumps(source.body["messages"], ensure_ascii=False)
            assert "分段分析" in joined
            assert "第 1 段" in joined
            assert "第 2 段" in joined
        finally:
            cfg.image.long_screenshot_ocr = old
            _clear_image_cache()

    def test_square_image_not_sliced(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old = cfg.image.long_screenshot_ocr
        cfg.image.long_screenshot_ocr = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "看这张图"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            assert len(vision.post_calls) == 1, "普通图片不应切片"
        finally:
            cfg.image.long_screenshot_ocr = old
            _clear_image_cache()


class TestVisionMaxTokens:
    """视觉渠道 max_tokens 配置生效。"""

    def test_payload_uses_configured_max_tokens(self):
        from app.vision import _call_vision_model
        from app.config import get_config
        cfg = get_config()
        provider = cfg.vision_providers["openai"]
        old = provider.max_tokens
        provider.max_tokens = 8192
        try:
            vision = _FakeVisionClient()
            with patch("app.vision.get_vision_client", return_value=vision):
                import asyncio
                asyncio.run(_call_vision_model(
                    provider_id="openai", model_id="", system_prompt="s",
                    user_question="q", image_urls=[TINY_PNG],
                ))
            assert vision.post_calls[0]["max_tokens"] == 8192
        finally:
            provider.max_tokens = old


class TestStructuredEvidence:
    """structured_evidence：视觉 JSON 证据解析、格式化、注入。"""

    def test_structure_formats_json_evidence(self):
        from app.vision import _structure_evidence
        json_text = json.dumps({
            "summary": "一张登录页截图",
            "ocr": {"full_text": "用户名\n密码\n登录"},
            "layout": {"regions": [
                {"type": "form", "reading_order": 1, "text": "用户名输入框"},
                {"type": "button", "reading_order": 2, "text": "登录"},
            ]},
            "semantics": {"scene": "登录页", "entities": [
                {"name": "登录按钮", "type": "button", "evidence": "底部"},
            ]},
            "visual": {"dominant_colors": ["#ffffff"], "style": "简洁"},
            "uncertainty": ["密码框文字模糊"],
        }, ensure_ascii=False)
        formatted = _structure_evidence(json_text)
        assert formatted is not None
        assert "【摘要】" in formatted and "登录页" in formatted
        assert "【全文文字】" in formatted and "用户名" in formatted
        assert "【版面结构】" in formatted
        assert "【关键实体】" in formatted and "登录按钮" in formatted
        assert "【不确定项 ⚠️】" in formatted and "模糊" in formatted

    def test_structure_tolerates_markdown_fence(self):
        from app.vision import _structure_evidence
        fenced = '```json\n{"summary": "图", "ocr": {"full_text": "abc"}, "uncertainty": []}\n```'
        formatted = _structure_evidence(fenced)
        assert formatted is not None and "abc" in formatted

    def test_structure_returns_none_for_invalid(self):
        from app.vision import _structure_evidence
        assert _structure_evidence("这不是 JSON") is None
        assert _structure_evidence("") is None
        assert _structure_evidence("普通描述文字") is None

    def test_structured_evidence_injected_end_to_end(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = cfg.image.structured_evidence
        cfg.image.structured_evidence = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这是什么页面?"},
                ])],
                stream=False,
            )
            evidence_json = json.dumps({
                "summary": "一个登录页面",
                "ocr": {"full_text": "欢迎回来"},
                "uncertainty": ["右上角图标不清晰"],
            }, ensure_ascii=False)
            vision = _ScriptedVisionClient([evidence_json])
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            joined = json.dumps(source.body["messages"], ensure_ascii=False)
            assert "【摘要】" in joined
            assert "【不确定项 ⚠️】" in joined
            assert "欢迎回来" in joined
        finally:
            cfg.image.structured_evidence = old
            _clear_image_cache()


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return False


class _FakeStreamResp:
    status_code = 200

    def __init__(self, lines):
        self._lines = list(lines)

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln


class _FakeStreamingSource:
    """模拟源模型流式 SSE 响应（思考链 + 回答 + usage + DONE）。"""

    def __init__(self):
        self.body = None
        self.lines = [
            'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"源模型思考中"}}]}',
            'data: {"choices":[{"delta":{"content":"源模型回答"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
            'data: [DONE]',
        ]

    def stream(self, method, url, json=None):
        self.body = json
        return _FakeStreamCtx(_FakeStreamResp(self.lines))


async def _fake_vision_stream(provider_id, model_id, system_prompt, user_question,
                              image_urls, request_client=None, emit=None):
    """模拟视觉模型流式输出：先 reasoning 后 content。"""
    await emit("reasoning", "我先分析这张图片…")
    await emit("content", "这是分析结果正文")
    return {"result": "这是分析结果正文", "model": "m", "elapsed": 0, "token_usage": {}}


class TestStreamVisionThinking:
    """流式视觉思考链：首帧即真实思考（默认无预提示），reasoning/content 都进思考链，
    再无缝衔接源模型思考链与回答。"""

    def _collect(self, prelude, cfg):
        from app.proxy import _combined_stream
        from app.image_utils import extract_all_images_with_positions, extract_current_turn_positions
        request = ChatCompletionRequest(
            model="openai-vision",
            messages=[ChatMessage(role="user", content=[
                {"type": "image_url", "image_url": {"url": TINY_PNG}},
                {"type": "text", "text": "这张图是什么?"},
            ])],
            stream=True,
        )
        raw = [m.model_dump(exclude_none=True) for m in request.messages]
        imgs = extract_all_images_with_positions(raw)
        cur = extract_current_turn_positions(raw)
        source = _FakeStreamingSource()
        old = cfg.image.vision_stream_prelude
        old_stream = cfg.image.stream_vision_thinking
        cfg.image.vision_stream_prelude = prelude
        cfg.image.stream_vision_thinking = True
        try:
            import asyncio

            async def collect():
                frames = []
                async for frame in _combined_stream(
                    request, cfg.models["openai-vision"], raw,
                    [i for i in imgs if i.position in cur], cur, _RawReq(),
                ):
                    frames.append(frame)
                return frames

            with patch("app.vision._call_vision_model_stream", new=_fake_vision_stream), \
                 patch("app.proxy.get_source_client", return_value=source):
                return asyncio.run(collect())
        finally:
            cfg.image.vision_stream_prelude = old
            cfg.image.stream_vision_thinking = old_stream

    def test_first_frame_is_real_reasoning(self):
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        try:
            frames = self._collect(prelude=False, cfg=cfg)
            all_text = "".join(frames)
            assert "正在处理" not in all_text, "默认不应有占位预提示"
            assert "我先分析这张图片" in all_text, "视觉模型 reasoning 应进入思考链"
            assert "这是分析结果正文" in all_text, "视觉模型 content 应进入思考链"
            assert "源模型思考中" in all_text, "源模型思考链应透传"
            assert "源模型回答" in all_text
            # 首帧就是视觉模型的真实思考链
            first_data = json.loads(frames[0][6:].strip())
            delta = first_data["choices"][0]["delta"]
            assert delta.get("role") == "assistant"
            assert "我先分析这张图片" in delta.get("reasoning_content", "")
        finally:
            _clear_image_cache()

    def test_prelude_optional(self):
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        try:
            frames = self._collect(prelude=True, cfg=cfg)
            first_data = json.loads(frames[0][6:].strip())
            assert "正在处理" in first_data["choices"][0]["delta"].get("reasoning_content", "")
        finally:
            _clear_image_cache()


class TestChannelNote:
    """vision_channel_note：注入通道说明，引导源模型按需重看图片。"""

    def test_channel_note_injected_when_enabled(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old = cfg.image.vision_channel_note
        cfg.image.vision_channel_note = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            joined = json.dumps(source.body["messages"], ensure_ascii=False)
            assert "不是图片本身" in joined, "应注入通道说明"
            assert "重新发送图片" in joined, "通道说明应引导重发图片触发重分析"
        finally:
            cfg.image.vision_channel_note = old
            _clear_image_cache()

    def test_channel_note_off_by_default(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old = cfg.image.vision_channel_note
        cfg.image.vision_channel_note = False
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            joined = json.dumps(source.body["messages"], ensure_ascii=False)
            assert "重新发送图片" not in joined
        finally:
            cfg.image.vision_channel_note = old
            _clear_image_cache()


class _ScriptedToolSource:
    """源模型：第一次返回 grassvision_view_image 工具调用，第二次返回正常回答。"""

    def __init__(self):
        self.bodies = []
        self.calls = 0

    async def post(self, url, json=None):
        self.bodies.append(json)
        import json as _json
        self.calls += 1
        if self.calls == 1:
            return _FakeResp(200, {"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "grassvision_view_image",
                    "arguments": _json.dumps({"question": "按钮是什么颜色"})}}],
            }}], "usage": {}})
        return _FakeResp(200, {"choices": [{"message": {"role": "assistant", "content": "按钮是蓝色渐变。"}}], "usage": {}})

    async def aclose(self):
        pass


class TestReexamineTool:
    """协议化服务端重看：源模型自主调用 view_image，服务端执行重看后喂回再回答。"""

    def test_service_side_reexamine_loop(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = cfg.image.vision_reexamine
        cfg.image.vision_reexamine = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这个按钮什么颜色?"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _ScriptedToolSource()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))
            data = json.loads(resp.body)
            assert data["choices"][0]["message"]["content"] == "按钮是蓝色渐变。"
            assert source.calls == 2, "源模型应被调用两次（工具循环）"
            # 第一次请求应注入工具
            tools = source.bodies[0].get("tools", [])
            names = [t.get("function", {}).get("name") for t in tools]
            assert "grassvision_view_image" in names, "应注入 view_image 工具"
            # 第二次请求应含 tool 结果（服务端已执行重看）
            tool_msgs = [m for m in source.bodies[1]["messages"] if m.get("role") == "tool"]
            assert len(tool_msgs) == 1, "应有 1 条 tool 结果消息"
            assert tool_msgs[0]["tool_call_id"] == "call_1"
            assert len(vision.post_calls) == 2, "首次分析 + 重看各 1 次视觉调用"
        finally:
            cfg.image.vision_reexamine = old
            _clear_image_cache()

    def test_no_tool_injected_when_disabled(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = cfg.image.vision_reexamine
        cfg.image.vision_reexamine = False
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图是什么?"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            tools = source.body.get("tools") or []
            names = [t.get("function", {}).get("name") for t in tools]
            assert "grassvision_view_image" not in names, "关闭时不应注入工具"
        finally:
            cfg.image.vision_reexamine = old
            _clear_image_cache()


class _ScriptedToolStreamSource:
    """流式源模型：第一轮流输出 view_image 工具增量，第二轮流输出正常回答。"""

    def __init__(self):
        self.bodies = []
        self.calls = 0

    def stream(self, method, url, json=None):
        self.bodies.append(json)
        self.calls += 1
        if self.calls == 1:
            lines = [
                'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_s1","type":"function","function":{"name":"grassvision_view_image","arguments":""}}]}}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"question\\":\\"颜色\\"}"}}]}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                'data: [DONE]',
            ]
        else:
            lines = [
                'data: {"choices":[{"delta":{"role":"assistant","content":"最终回答"}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
                'data: [DONE]',
            ]
        return _FakeStreamCtx(_FakeStreamResp(lines))


class TestStreamReexamine:
    """流式服务端重看：吞掉工具轮，客户端只看到最终回答流。"""

    def test_stream_swallows_tool_round(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old_r = cfg.image.vision_reexamine
        old_s = cfg.image.stream_vision_thinking
        cfg.image.vision_reexamine = True
        cfg.image.stream_vision_thinking = False
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这个按钮什么颜色?"},
                ])],
                stream=True,
            )
            vision = _FakeVisionClient()
            source = _ScriptedToolStreamSource()
            # 重看走流式视觉调用（stream_vision 恒 True），需同时 patch 流式视觉
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.vision._call_vision_model_stream", new=_fake_vision_stream), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))

                async def collect():
                    chunks = []
                    async for f in resp.body_iterator:
                        chunks.append(f)
                    return chunks

                frames = asyncio.run(collect())
            all_text = "".join(frames)
            assert "grassvision_view_image" not in all_text, "工具帧不应泄漏给客户端"
            assert "tool_calls" not in all_text
            assert "最终回答" in all_text, "重看后的最终回答应到达客户端"
            assert source.calls == 2, "源模型应被调用两次（工具轮被吞后重发）"
            tool_msgs = [m for m in source.bodies[1]["messages"] if m.get("role") == "tool"]
            assert len(tool_msgs) == 1, "第二轮应携带工具结果"
            assert len(vision.post_calls) == 1, "首次分析走非流式（stream_vision_thinking 关）"
            # 重看的思考链应透传给客户端（stream_vision 恒 True）
            assert all_text.count("这是分析结果正文") >= 1, "重看思考链应流式透传"
        finally:
            cfg.image.vision_reexamine = old_r
            cfg.image.stream_vision_thinking = old_s
            _clear_image_cache()

    def test_client_own_tools_passthrough(self):
        """客户端自己的工具调用应原样透传（不影响 harness 自带工具）。"""
        from app.proxy import _collect_tool_calls, _assistant_msg_with_tool_calls, _strip_grassvision_tools
        deltas = [
            {"index": 0, "id": "c1", "type": "function", "function": {"name": "browse", "arguments": ""}},
            {"index": 0, "function": {"arguments": "{\"url\":\"x\"}"}},
        ]
        calls = _collect_tool_calls(deltas)
        assert calls[0]["function"]["name"] == "browse"
        msg = _assistant_msg_with_tool_calls(calls)
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"url":"x"}'
        # 移除注入工具不影响客户端工具
        body = {"tools": [{"type": "function", "function": {"name": "browse"}},
                          {"type": "function", "function": {"name": "grassvision_view_image"}}]}
        stripped = _strip_grassvision_tools(body)
        names = [t["function"]["name"] for t in stripped["tools"]]
        assert names == ["browse"]


class TestFusedStream:
    """融合版：视觉思考流 + 流式服务端重看同时工作。"""

    def test_fused_vision_thinking_and_reexamine(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old_r = cfg.image.vision_reexamine
        old_s = cfg.image.stream_vision_thinking
        old_p = cfg.image.vision_stream_prelude
        cfg.image.vision_reexamine = True
        cfg.image.stream_vision_thinking = True
        cfg.image.vision_stream_prelude = False
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这个按钮什么颜色?"},
                ])],
                stream=True,
            )
            source = _ScriptedToolStreamSource()
            with patch("app.vision._call_vision_model_stream", new=_fake_vision_stream), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))

                async def collect():
                    chunks = []
                    async for f in resp.body_iterator:
                        chunks.append(f)
                    return chunks

                frames = asyncio.run(collect())
            all_text = "".join(frames)
            # 阶段 1 视觉思考可见
            assert "我先分析这张图片" in all_text, "阶段1视觉思考应推给客户端"
            # 工具轮被吞、无泄漏
            assert "grassvision_view_image" not in all_text
            assert "tool_calls" not in all_text
            # 重看视觉思考② + 最终回答
            assert all_text.count("这是分析结果正文") >= 2, "阶段1分析 + 重看分析都应进入思考链"
            assert "最终回答" in all_text
            assert source.calls == 2
        finally:
            cfg.image.vision_reexamine = old_r
            cfg.image.stream_vision_thinking = old_s
            cfg.image.vision_stream_prelude = old_p
            _clear_image_cache()


class TestCrossTurnReexamine:
    """跨轮次无感重看：第二轮无当前图片时，仍注入工具允许重看历史图片。"""

    def test_no_image_turn_injects_tool(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = cfg.image.vision_reexamine
        cfg.image.vision_reexamine = True
        try:
            # 第二轮：历史有图（被剥离），当前轮纯文字追问
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[
                    ChatMessage(role="user", content=[
                        {"type": "image_url", "image_url": {"url": TINY_PNG}},
                        {"type": "text", "text": "看这张图"},
                    ]),
                    ChatMessage(role="assistant", content="看到了。"),
                    ChatMessage(role="user", content="刚才图里的按钮什么颜色?"),
                ],
                stream=False,
            )
            vision = _FakeVisionClient()
            source = _FakeSourceClient()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                asyncio.run(handle_chat_completion(request, _RawReq()))
            tools = source.body.get("tools") or []
            names = [t.get("function", {}).get("name") for t in tools]
            assert "grassvision_view_image" in names, "无当前图片时也应注入重看工具（历史图可重看）"
        finally:
            cfg.image.vision_reexamine = old
            _clear_image_cache()


class TestUsageAggregation:
    """重看用量聚合 + 缓存命中率口径。"""

    def test_reexamine_aggregates_source_usage(self):
        from app.proxy import _forward_with_reexamine
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        model = cfg.models["openai-vision"]
        old = cfg.image.vision_reexamine
        cfg.image.vision_reexamine = True
        try:
            # 源客户端：第一轮 tool_call（usage 5），第二轮回答（usage 10）
            class _AggSource:
                def __init__(self):
                    self.calls = 0

                async def post(self, url, json=None):
                    import json as _json
                    self.calls += 1
                    if self.calls == 1:
                        return _FakeResp(200, {"choices": [{"message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [{"id": "c1", "type": "function", "function": {
                                "name": "grassvision_view_image",
                                "arguments": _json.dumps({"question": "颜色"})}}],
                        }}], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})
                    return _FakeResp(200, {"choices": [{"message": {"role": "assistant", "content": "蓝色"}}],
                                           "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}})

                async def aclose(self):
                    pass

            source = _AggSource()
            vision = _FakeVisionClient()
            body = {"model": "openai-vision", "messages": [{"role": "user", "content": "看这张图"}]}
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp, agg_source, agg_vision = asyncio.run(_forward_with_reexamine(
                    body=body, provider_key="openai",
                    public_model_id="openai-vision", model=model,
                    images=[],
                    raw_request=type("R", (), {})(),
                ))
            assert agg_source.get("prompt_tokens") == 15, "多轮 source usage 应聚合"
            assert agg_source.get("total_tokens") == 25
            assert source.calls == 2
        finally:
            cfg.image.vision_reexamine = old
            _clear_image_cache()


class TestChannelNoteModes:
    """通道说明按 reexamine 是否开启动态切换（工具版/重发版）。"""

    def test_channel_note_with_tool_when_reexamine_on(self):
        from app.proxy import _inject_channel_note, _CHANNEL_NOTE_TEXT, _CHANNEL_NOTE_TEXT_NO_TOOL
        out = _inject_channel_note([{"role": "system", "content": "base"}], with_tool=True)
        assert "grassvision_view_image" in out[0]["content"]
        out2 = _inject_channel_note([{"role": "system", "content": "base"}], with_tool=False)
        assert "grassvision_view_image" not in out2[0]["content"]
        assert "重新发送图片" in out2[0]["content"]
        assert _CHANNEL_NOTE_TEXT != _CHANNEL_NOTE_TEXT_NO_TOOL


class TestCrossTurnStreamReexamineThinking:
    """跨轮次（无当前图片）流式重看：思考链必须透传。"""

    def test_cross_turn_stream_thinking_passthrough(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old_r = cfg.image.vision_reexamine
        old_s = cfg.image.stream_vision_thinking
        cfg.image.vision_reexamine = True
        cfg.image.stream_vision_thinking = False  # 首次不流式，仅验证重看思考链
        try:
            # 第二轮：历史有图（当前轮无图），流式追问
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[
                    ChatMessage(role="user", content=[
                        {"type": "image_url", "image_url": {"url": TINY_PNG}},
                        {"type": "text", "text": "看这张图"},
                    ]),
                    ChatMessage(role="assistant", content="看到了。"),
                    ChatMessage(role="user", content="按钮什么颜色?"),
                ],
                stream=True,
            )
            vision = _FakeVisionClient()
            source = _ScriptedToolStreamSource()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.vision._call_vision_model_stream", new=_fake_vision_stream), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))

                async def collect():
                    chunks = []
                    async for f in resp.body_iterator:
                        chunks.append(f)
                    return chunks

                frames = asyncio.run(collect())
            all_text = "".join(frames)
            assert "grassvision_view_image" not in all_text, "工具帧不应泄漏"
            assert "这是分析结果正文" in all_text, "跨轮重看的思考链应流式透传"
            assert "最终回答" in all_text
        finally:
            cfg.image.vision_reexamine = old_r
            cfg.image.stream_vision_thinking = old_s
            _clear_image_cache()


class TestToolConflictGuard:
    """客户端自带工具时不注入 grassvision 工具；混合工具轮整轮透传。"""

    def test_inject_even_when_client_has_tools(self):
        """客户端自带工具时也要追加 grassvision 工具（追加是核心功能）。"""
        from app.proxy import _inject_grassvision_tools
        body = {"tools": [{"type": "function", "function": {"name": "client_tool"}}]}
        out = _inject_grassvision_tools(body, reexamine=True, pixel_tools=True)
        names = [t["function"]["name"] for t in out["tools"]]
        assert "client_tool" in names, "客户端工具保留"
        assert "grassvision_view_image" in names, "应追加 grassvision 工具"
        assert "grassvision_pixel_colors" in names

    def test_inject_when_no_client_tools(self):
        from app.proxy import _inject_grassvision_tools
        body = {"messages": []}
        out = _inject_grassvision_tools(body, reexamine=True, pixel_tools=True)
        names = [t["function"]["name"] for t in out.get("tools", [])]
        assert "grassvision_view_image" in names
        assert "grassvision_pixel_colors" in names

    def test_mixed_tool_round_full_closure(self):
        """混合工具调用（grassvision + 客户端工具）→ 服务端完整闭环：
        每条 tool_call_id 都有 tool 消息（grassvision 真实结果 + 客户端工具占位说明），
        最终响应不泄漏 tool_calls。"""
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old = cfg.image.vision_reexamine
        cfg.image.vision_reexamine = True
        try:
            class _MixedSource:
                def __init__(self):
                    self.calls = 0
                    self.bodies = []

                async def post(self, url, json=None):
                    self.bodies.append(json)
                    self.calls += 1
                    if self.calls == 1:
                        return _FakeResp(200, {"choices": [{"message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "grassvision_view_image", "arguments": '{"question":"颜色"}'}},
                                {"id": "c2", "type": "function", "function": {"name": "client_tool", "arguments": "{}"}},
                            ]}}], "usage": {}})
                    return _FakeResp(200, {"choices": [{"message": {"role": "assistant", "content": "图片已分析。"}}], "usage": {}})

                async def aclose(self):
                    pass

            source = _MixedSource()
            vision = _FakeVisionClient()
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "看这张图"},
                ])],
                stream=False,
            )
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))
            data = json.loads(resp.body)
            msg = data["choices"][0]["message"]
            assert not msg.get("tool_calls"), "混合工具轮应服务端闭环，不泄漏 tool_calls"
            assert source.calls == 2, "混合工具轮应进入服务端工具循环（执行后二次调用源模型）"
            # 第二次请求的 tool 消息应覆盖全部 tool_call_id（含客户端工具占位）
            tool_msgs = [m for m in source.bodies[1]["messages"] if m.get("role") == "tool"]
            assert len(tool_msgs) == 2, "两条 tool_call 都应有响应"
            ids = [m["tool_call_id"] for m in tool_msgs]
            assert "c1" in ids and "c2" in ids, "所有 tool_call_id 都应有响应（不悬空）"
        finally:
            cfg.image.vision_reexamine = old
            _clear_image_cache()


class TestReasoningPassthrough:
    """流式吞掉工具轮时，assistant 消息必须回传 reasoning_content（thinking 模式）。"""

    def test_swallowed_tool_round_keeps_reasoning(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old_r = cfg.image.vision_reexamine
        old_s = cfg.image.stream_vision_thinking
        cfg.image.vision_reexamine = True
        cfg.image.stream_vision_thinking = False
        try:
            class _ReasoningToolSource:
                def __init__(self):
                    self.bodies = []
                    self.calls = 0

                def stream(self, method, url, json=None):
                    self.bodies.append(json)
                    self.calls += 1
                    if self.calls == 1:
                        lines = [
                            'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"我需要重新看这张图…"}}]}',
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_r1","type":"function","function":{"name":"grassvision_view_image","arguments":""}}]}}]}',
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"question\\":\\"颜色\\"}"}}]}}]}',
                            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                            'data: [DONE]',
                        ]
                    else:
                        lines = [
                            'data: {"choices":[{"delta":{"role":"assistant","content":"最终回答"}}]}',
                            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                            'data: [DONE]',
                        ]
                    return _FakeStreamCtx(_FakeStreamResp(lines))

            source = _ReasoningToolSource()
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这个按钮什么颜色?"},
                ])],
                stream=True,
            )
            with patch("app.vision._call_vision_model_stream", new=_fake_vision_stream), \
                 patch("app.vision.get_vision_client", return_value=_FakeVisionClient()), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))

                async def collect():
                    chunks = []
                    async for f in resp.body_iterator:
                        chunks.append(f)
                    return chunks

                frames = asyncio.run(collect())
            all_text = "".join(frames)
            assert "最终回答" in all_text
            # 第二轮请求的 assistant 消息必须包含 reasoning_content（thinking 模式回传）
            assistant_msgs = [m for m in source.bodies[1]["messages"] if m.get("role") == "assistant"]
            tool_assistant = [m for m in assistant_msgs if m.get("tool_calls")]
            assert tool_assistant, "第二轮应有带 tool_calls 的 assistant 消息"
            assert "我需要重新看这张图" in tool_assistant[-1].get("reasoning_content", ""), \
                f"吞轮时的 reasoning_content 必须回传，实际 {tool_assistant[-1].get('reasoning_content', '')!r}"
        finally:
            cfg.image.vision_reexamine = old_r
            cfg.image.stream_vision_thinking = old_s
            _clear_image_cache()


class TestStreamFinishReason:
    """流异常中断（上游未发 [DONE]）时，必须补 finish_reason + [DONE]。"""

    def test_interrupted_stream_gets_finish_reason(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        _clear_image_cache()
        cfg = get_config()
        old_r = cfg.image.vision_reexamine
        old_s = cfg.image.stream_vision_thinking
        cfg.image.vision_reexamine = True
        cfg.image.stream_vision_thinking = False
        try:
            class _InterruptSource:
                def __init__(self):
                    self.calls = 0

                def stream(self, method, url, json=None):
                    self.calls += 1
                    if self.calls == 1:
                        # 工具轮（无 [DONE]，正常：工具轮被吞）
                        lines = [
                            'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"思考中…"}}]}',
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"grassvision_view_image","arguments":"{}"}}]}}]}',
                            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                        ]
                    else:
                        # 第二轮流：只发思考帧，然后自然结束（无 [DONE]）→ 上游断流模拟
                        lines = [
                            'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"思考特别长…"}}]}',
                            'data: {"choices":[{"delta":{"content":"回答"}}]}',
                        ]
                    return _FakeStreamCtx(_FakeStreamResp(lines))

            source = _InterruptSource()
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图?"},
                ])],
                stream=True,
            )
            with patch("app.vision._call_vision_model_stream", new=_fake_vision_stream), \
                 patch("app.vision.get_vision_client", return_value=_FakeVisionClient()), \
                 patch("app.proxy.get_source_client", return_value=source):
                import asyncio
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))

                async def collect():
                    chunks = []
                    async for f in resp.body_iterator:
                        chunks.append(f)
                    return chunks

                frames = asyncio.run(collect())
            all_text = "".join(frames)
            assert '"finish_reason": "stop"' in all_text, "流中断必须补 finish_reason 帧"
            assert all_text.rstrip().endswith("data: [DONE]"), f"流必须以 [DONE] 结尾，实际末尾 {all_text[-40:]!r}"
        finally:
            cfg.image.vision_reexamine = old_r
            cfg.image.stream_vision_thinking = old_s
            _clear_image_cache()
