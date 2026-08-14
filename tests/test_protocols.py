"""协议适配层测试：Anthropic Messages 转换 + 端到端。"""
import asyncio
import base64
import io
import json
from unittest.mock import patch

from PIL import Image

from app.config import get_config
from app.protocols.anthropic import parse_messages_request, build_messages_response
from app.schemas import ChatMessage


def _png_b64(rgb=(37, 99, 235), size=(64, 64)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestParseMessages:
    def test_system_and_text_user(self):
        req = parse_messages_request({
            "model": "openai-vision",
            "system": "你是助手",
            "messages": [{"role": "user", "content": "你好"}],
        })
        assert req.messages[0].role == "system"
        assert req.messages[0].content == "你是助手"
        assert req.messages[1].role == "user"
        assert req.messages[1].content == "你好"

    def test_image_block_to_data_url(self):
        b64 = _png_b64()
        req = parse_messages_request({
            "model": "openai-vision",
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": "看这张图"},
            ]}],
        })
        msg = req.messages[0]
        assert isinstance(msg.content, list)
        assert msg.content[0]["type"] == "image_url"
        assert msg.content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert msg.content[1]["type"] == "text"

    def test_tool_use_and_tool_result_roundtrip(self):
        req = parse_messages_request({
            "model": "openai-vision",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "我查一下"},
                    {"type": "tool_use", "id": "tu1", "name": "bash", "input": {"command": "ls"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "file.txt"},
                ]},
            ],
        })
        assert req.messages[0].role == "assistant"
        assert req.messages[0].tool_calls is not None
        assert req.messages[0].tool_calls[0]["function"]["name"] == "bash"
        args = json.loads(req.messages[0].tool_calls[0]["function"]["arguments"])
        assert args == {"command": "ls"}
        assert req.messages[1].role == "tool"
        assert req.messages[1].tool_call_id == "tu1"
        assert req.messages[1].content == "file.txt"

    def test_tools_conversion(self):
        req = parse_messages_request({
            "model": "openai-vision",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "bash", "description": "run", "input_schema": {"type": "object", "properties": {}}}],
        })
        assert req.tools[0]["type"] == "function"
        assert req.tools[0]["function"]["name"] == "bash"
        assert req.tools[0]["function"]["parameters"]["type"] == "object"


class TestBuildMessagesResponse:
    def test_text_response(self):
        out = build_messages_response({
            "choices": [{"message": {"role": "assistant", "content": "蓝色按钮"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }, "openai-vision")
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["content"][0]["type"] == "text"
        assert out["content"][0]["text"] == "蓝色按钮"
        assert out["stop_reason"] == "end_turn"
        assert out["usage"]["input_tokens"] == 10

    def test_tool_use_response(self):
        out = build_messages_response({
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "bash", "arguments": '{"command":"ls"}'}}],
            }}],
            "usage": {},
        }, "openai-vision")
        assert out["stop_reason"] == "tool_use"
        tu = out["content"][0]
        assert tu["type"] == "tool_use"
        assert tu["name"] == "bash"
        assert tu["input"] == {"command": "ls"}


class TestAnthropicEndpoint:
    """端到端：/v1/messages 请求 → 核心处理 → Anthropic 响应。"""

    def test_non_stream_end_to_end(self):
        from app.main import app
        from fastapi.testclient import TestClient
        from tests.test_proxy import _FakeVisionClient, _FakeSourceClient

        b64 = _png_b64()
        cfg = get_config()
        old_r = cfg.image.vision_reexamine
        cfg.image.vision_reexamine = True
        try:
            with TestClient(app) as client, \
                 patch("app.vision.get_vision_client", return_value=_FakeVisionClient()), \
                 patch("app.proxy.get_source_client", return_value=_FakeSourceClient()):
                resp = client.post("/v1/messages", json={
                    "model": "openai-vision",
                    "system": "你是图像分析助手",
                    "messages": [{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": "这张图什么颜色?"},
                    ]}],
                })
            assert resp.status_code == 200, resp.text[:200]
            data = resp.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert data["content"][0]["type"] == "text"
            assert data["content"][0]["text"]
            assert "usage" in data
        finally:
            cfg.image.vision_reexamine = old_r

    def test_stream_end_to_end(self):
        from app.main import app
        from fastapi.testclient import TestClient
        from tests.test_proxy import _FakeVisionClient, _FakeStreamingSource

        b64 = _png_b64()
        cfg = get_config()
        old_s = cfg.image.stream_vision_thinking
        cfg.image.stream_vision_thinking = False
        try:
            with TestClient(app) as client, \
                 patch("app.vision.get_vision_client", return_value=_FakeVisionClient()), \
                 patch("app.proxy.get_source_client", return_value=_FakeStreamingSource()):
                with client.stream("POST", "/v1/messages", json={
                    "model": "openai-vision",
                    "messages": [{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": "这张图什么颜色?"},
                    ]}],
                    "stream": True,
                }) as resp:
                    body = resp.read().decode("utf-8")
                assert resp.status_code == 200, body[:200]
            assert "event: message_start" in body, "流应以 message_start 开头"
            assert "event: content_block_delta" in body, "应有文本增量"
            assert "event: message_stop" in body, "流应以 message_stop 结尾"
            assert "text_delta" in body
        finally:
            cfg.image.stream_vision_thinking = old_s


class TestResponses:
    """Responses API：解析 + 序列化 + 端到端。"""

    def test_parse_input_items(self):
        from app.protocols.responses import parse_responses_request
        req = parse_responses_request({
            "model": "openai-vision",
            "input": [
                {"role": "system", "content": "助手"},
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                    {"type": "input_text", "text": "看这张图"},
                ]},
                {"type": "function_call", "call_id": "fc1", "name": "bash", "arguments": '{"command":"ls"}'},
                {"type": "function_call_output", "call_id": "fc1", "output": "file.txt"},
            ],
            "tools": [{"type": "function", "name": "bash", "description": "run", "parameters": {"type": "object"}}],
        })
        assert req.messages[0].role == "system"
        assert req.messages[1].content[0]["type"] == "image_url"
        assert req.messages[2].tool_calls[0]["function"]["name"] == "bash"
        assert req.messages[3].role == "tool"
        assert req.messages[3].tool_call_id == "fc1"
        assert req.tools[0]["function"]["name"] == "bash"

    def test_build_response(self):
        from app.protocols.responses import build_responses_response
        out = build_responses_response({
            "choices": [{"message": {"role": "assistant", "content": "蓝色"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        }, "openai-vision")
        assert out["object"] == "response"
        assert out["status"] == "completed"
        assert out["output"][0]["type"] == "message"
        assert out["output"][0]["content"][0]["type"] == "output_text"
        assert out["output"][0]["content"][0]["text"] == "蓝色"

    def test_non_stream_end_to_end(self):
        from app.main import app
        from fastapi.testclient import TestClient
        from tests.test_proxy import _FakeVisionClient, _FakeSourceClient

        b64 = _png_b64()
        cfg = get_config()
        old_r = cfg.image.vision_reexamine
        cfg.image.vision_reexamine = True
        try:
            with TestClient(app) as client, \
                 patch("app.vision.get_vision_client", return_value=_FakeVisionClient()), \
                 patch("app.proxy.get_source_client", return_value=_FakeSourceClient()):
                resp = client.post("/v1/responses", json={
                    "model": "openai-vision",
                    "input": [{"role": "user", "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                        {"type": "input_text", "text": "这张图什么颜色?"},
                    ]}],
                })
            assert resp.status_code == 200, resp.text[:200]
            data = resp.json()
            assert data["object"] == "response"
            assert data["output"][0]["content"][0]["type"] == "output_text"
            assert data["output"][0]["content"][0]["text"]
        finally:
            cfg.image.vision_reexamine = old_r

    def test_stream_end_to_end(self):
        from app.main import app
        from fastapi.testclient import TestClient
        from tests.test_proxy import _FakeVisionClient, _FakeStreamingSource

        b64 = _png_b64()
        cfg = get_config()
        old_s = cfg.image.stream_vision_thinking
        cfg.image.stream_vision_thinking = False
        try:
            with TestClient(app) as client, \
                 patch("app.vision.get_vision_client", return_value=_FakeVisionClient()), \
                 patch("app.proxy.get_source_client", return_value=_FakeStreamingSource()):
                with client.stream("POST", "/v1/responses", json={
                    "model": "openai-vision",
                    "input": [{"role": "user", "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                        {"type": "input_text", "text": "这张图什么颜色?"},
                    ]}],
                    "stream": True,
                }) as resp:
                    body = resp.read().decode("utf-8")
                assert resp.status_code == 200, body[:200]
            assert "event: response.created" in body
            assert "event: response.output_text.delta" in body
            assert "event: response.completed" in body
        finally:
            cfg.image.stream_vision_thinking = old_s
