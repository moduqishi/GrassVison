"""本地像素工具测试：算法精度 + 服务端无感拦截执行。"""
import asyncio
import base64
import io
import json

import pytest
from unittest.mock import patch
from PIL import Image, ImageDraw

from app import pixel_tools as PT
from app.config import get_config
from app.proxy import handle_chat_completion
from app.schemas import ChatCompletionRequest, ChatMessage
from app.image_cache import get_image_cache


def _make_png(rgb, size=(100, 100)):
    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


TINY_PNG = _make_png((37, 99, 235), (120, 120))  # 纯蓝色 #2563EB


class TestDominantColors:
    def test_exact_color_of_solid_image(self):
        raw = base64.b64decode(TINY_PNG.split(",", 1)[1])
        colors = PT.dominant_colors(raw, top=3)
        assert colors and colors[0]["color"] == "#2563EB", f"应为精确蓝色，实际 {colors}"
        assert colors[0]["share"] > 0.9

    def test_candidate_matching(self):
        raw = base64.b64decode(TINY_PNG.split(",", 1)[1])
        cands = ["#F3F4F6", "#2563EB", "#111827"]
        matched = PT.dominant_colors(raw, candidates=cands)
        matched_labels = [c["color"] for c in matched if c.get("matched")]
        assert matched_labels and matched_labels[0] == "#2563EB", f"最接近的候选应为蓝色，实际 {matched_labels}"


class TestPixelDiff:
    def test_identical_regions_zero_diff(self):
        raw = base64.b64decode(TINY_PNG.split(",", 1)[1])
        r = PT.pixel_diff(raw, "0,0,500,500", "500,0,1000,500")
        assert r["diff_percent"] == 0.0, "相同纯色区域差异应为 0"

    def test_different_colors_detect_diff(self):
        blue = base64.b64decode(TINY_PNG.split(",", 1)[1])
        red = base64.b64decode(_make_png((220, 38, 38)).split(",", 1)[1])  # #DC2626
        # 用同一张图的两个区域：蓝色 vs 红色 → 差异大
        r = PT.pixel_diff(blue, "0,0,500,500", "500,0,1000,500")
        assert r["diff_percent"] == 0.0
        # 构造左蓝右红的图
        img = Image.new("RGB", (200, 100), (37, 99, 235))
        for y in range(100):
            for x in range(100, 200):
                img.putpixel((x, y), (220, 38, 38))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw2 = buf.getvalue()
        r2 = PT.pixel_diff(raw2, "0,0,500,500", "500,0,1000,500")
        assert r2["diff_percent"] > 20, f"两色区域应差异明显，实际 {r2}"


class TestTrace:
    def test_solid_region_geometry(self):
        # 白底 + 中央蓝色矩形（前景 = 蓝色块）
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        for y in range(60, 140):
            for x in range(60, 140):
                img.putpixel((x, y), (37, 99, 235))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        geo = PT.trace_region(raw, "0,0,1000,1000")
        assert geo["dominant_color"] == "#2563EB"
        assert geo["width"] >= 70 and geo["height"] >= 70, f"前景矩形应约 80px，实际 {geo['width']}x{geo['height']}"
        assert geo["foreground_box_px"] is not None


class TestPixelToolsServiceSide:
    """服务端无感：源模型调 grassvision_pixel_colors → 服务端执行 → 返回精确色值。"""

    class _ToolSource:
        def __init__(self):
            self.calls = 0
            self.bodies = []

        async def post(self, url, json=None):
            self.bodies.append(json)
            self.calls += 1
            if self.calls == 1:
                return _FakeResp2(200, {"choices": [{"message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "grassvision_pixel_colors",
                        "arguments": '{"region": "0,0,1000,1000"}'}}],
                }}], "usage": {}})
            return _FakeResp2(200, {"choices": [{"message": {"role": "assistant", "content": "按钮是蓝色。"}}], "usage": {}})

        async def aclose(self):
            pass

    def test_pixel_colors_executed_server_side(self):
        from app.image_cache import get_image_cache
        _clear = get_image_cache()
        asyncio.run(_clear.clear())
        cfg = get_config()
        old = cfg.image.pixel_tools
        cfg.image.pixel_tools = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这个按钮什么颜色？"},
                ])],
                stream=False,
            )
            vision = _FakeVisionClient2()
            source = self._ToolSource()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                resp = asyncio.run(handle_chat_completion(request, type("R", (), {})()))
            import json as _json
            data = _json.loads(resp.body)
            assert data["choices"][0]["message"]["content"] == "按钮是蓝色。"
            assert source.calls == 2, "工具循环应执行"
            # 第二次请求应携带工具结果（精确色值）
            tool_msgs = [m for m in source.bodies[1]["messages"] if m.get("role") == "tool"]
            assert len(tool_msgs) == 1
            assert "#2563EB" in tool_msgs[0]["content"], f"工具结果应含精确色值，实际 {tool_msgs[0]['content'][:100]}"
            # 首次请求注入像素工具
            names = [t["function"]["name"] for t in source.bodies[0].get("tools", [])]
            assert "grassvision_pixel_colors" in names
        finally:
            cfg.image.pixel_tools = old
            asyncio.run(_clear.clear())


class _FakeResp2:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class _FakeVisionClient2:
    def __init__(self):
        self.post_calls = []

    async def post(self, url, json=None):
        self.post_calls.append(json)
        return _FakeResp2(200, {"choices": [{"message": {"content": "描述"}}], "usage": {}})

    async def aclose(self):
        pass


class TestCrossImageDiff:
    def test_two_different_images_diff(self):
        blue = base64.b64decode(TINY_PNG.split(",", 1)[1])
        red_raw = base64.b64decode(_make_png((220, 38, 38)).split(",", 1)[1])
        r = PT.pixel_diff_images(blue, red_raw)
        assert r["diff_percent"] > 20, f"蓝 vs 红应差异明显，实际 {r}"

    def test_identical_images_zero_diff(self):
        blue = base64.b64decode(TINY_PNG.split(",", 1)[1])
        r = PT.pixel_diff_images(blue, blue)
        assert r["diff_percent"] == 0.0


class TestRenderHtml:
    def test_render_simple_html(self):
        html = "<html><body style='background:#ff0000'><h1>Hello</h1></body></html>"
        shot = PT.render_html(html, width=200, height=150)
        assert shot[:8] == b"\x89PNG\r\n\x1a\n", "应渲染出 PNG"
        colors = PT.dominant_colors(shot, top=2)
        assert colors and colors[0]["color"] == "#FF0000", f"背景应为红，实际 {colors}"


class TestAutoPixelInject:
    def test_pixel_evidence_injected(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        from app.image_cache import get_image_cache
        _clear = get_image_cache()
        asyncio.run(_clear.clear())
        cfg = get_config()
        old = cfg.image.auto_pixel_inject
        cfg.image.auto_pixel_inject = True
        try:
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图什么颜色?"},
                ])],
                stream=False,
            )
            source = _SimpleSource()
            vision = _FakeVisionClient2()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                resp = asyncio.run(handle_chat_completion(request, type("R", (), {})()))
            import json as _json
            data = _json.loads(resp.body)
            content = data["choices"][0]["message"]["content"]
            assert content == "按钮是蓝色。"
            # 注入的描述应包含像素证据（精确色值）
            msgs = source.bodies[0]["messages"]
            joined = _json.dumps(msgs, ensure_ascii=False)
            assert "像素证据" in joined, "应自动注入像素证据"
            assert "#2563EB" in joined, f"注入的色值应精确，实际 {joined[-200:]}"
        finally:
            cfg.image.auto_pixel_inject = old
            asyncio.run(_clear.clear())


class _SimpleSource:
    def __init__(self):
        self.bodies = []

    async def post(self, url, json=None):
        self.bodies.append(json)
        return _FakeResp2(200, {"choices": [{"message": {"role": "assistant", "content": "按钮是蓝色。"}}], "usage": {}})

    async def aclose(self):
        pass


class TestAutoPixelInjectStreaming:
    """融合流（stream_vision_thinking 开启）下，首次分析也应自动注入主色。"""

    def test_streaming_fused_injects_pixel_evidence(self):
        from app.proxy import handle_chat_completion
        from app.config import get_config
        from app.image_cache import get_image_cache
        _clear = get_image_cache()
        asyncio.run(_clear.clear())
        cfg = get_config()
        old_inj = cfg.image.auto_pixel_inject
        old_s = cfg.image.stream_vision_thinking
        cfg.image.auto_pixel_inject = True
        cfg.image.stream_vision_thinking = True
        try:
            class _StreamSource:
                def __init__(self):
                    self.bodies = []

                def stream(self, method, url, json=None):
                    self.bodies.append(json)
                    lines = [
                        'data: {"choices":[{"delta":{"role":"assistant","content":"流式回答"}}]}',
                        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                        'data: [DONE]',
                    ]
                    return _FakeStreamCtx(_FakeStreamResp(lines))

            source = _StreamSource()
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "这张图什么颜色?"},
                ])],
                stream=True,
            )
            import sys
            sys.path.insert(0, "/Users/cake/toys/GrassVison")
            from tests.test_proxy import _RawReq, _FakeStreamCtx, _FakeStreamResp, _fake_vision_stream
            with patch("app.vision._call_vision_model_stream", new=_fake_vision_stream), \
                 patch("app.proxy.get_source_client", return_value=source):
                resp = asyncio.run(handle_chat_completion(request, _RawReq()))

                async def collect():
                    chunks = []
                    async for f in resp.body_iterator:
                        chunks.append(f)
                    return chunks

                asyncio.run(collect())
            # 融合流下注入的描述应含像素证据（精确色值）
            msgs = source.bodies[0]["messages"]
            joined = json.dumps(msgs, ensure_ascii=False)
            assert "像素证据" in joined, f"融合流首次分析应注入像素证据，实际 {joined[-150:]}"
            assert "#2563EB" in joined
        finally:
            cfg.image.auto_pixel_inject = old_inj
            cfg.image.stream_vision_thinking = old_s
            asyncio.run(_clear.clear())


class TestShapeRecognition:
    """图元识别：已知形状图 → 正确 SVG 图元。"""

    @staticmethod
    def _make_shapes_img():
        img = Image.new("RGB", (360, 220), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.ellipse((20, 20, 80, 80), fill=(37, 99, 235))
        d.rectangle((120, 30, 200, 70), fill=(37, 99, 235))
        d.line((30, 200, 330, 200), fill=(37, 99, 235), width=4)
        d.polygon([(260, 40), (320, 40), (290, 90)], fill=(37, 99, 235))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_recognizes_all_primitives(self):
        raw = self._make_shapes_img()
        r = PT.trace_region(raw)
        types = [s["type"] for s in r.get("shapes", [])]
        assert "circle" in types, f"应识别圆，实际 {types}"
        assert "rect" in types, f"应识别矩形，实际 {types}"
        assert "line" in types, f"应识别线，实际 {types}"
        assert "polygon" in types, f"应识别多边形，实际 {types}"

    def test_circle_geometry_accurate(self):
        raw = self._make_shapes_img()
        r = PT.trace_region(raw)
        circle = next(s for s in r["shapes"] if s["type"] == "circle")
        assert abs(circle["cx"] - 50) <= 3, f"圆心 x 应≈50，实际 {circle['cx']}"
        assert abs(circle["cy"] - 50) <= 3, f"圆心 y 应≈50，实际 {circle['cy']}"
        assert abs(circle["r"] - 30) <= 3, f"半径应≈30，实际 {circle['r']}"

    def test_svg_contains_primitives(self):
        raw = self._make_shapes_img()
        r = PT.trace_region(raw)
        svg = r.get("svg", "")
        assert "<circle " in svg
        assert "<rect " in svg
        assert "<line " in svg
        assert "<polygon " in svg or "<polyline " in svg

    def test_description_text(self):
        raw = self._make_shapes_img()
        r = PT.trace_region(raw)
        desc = r.get("description", "")
        assert "圆形" in desc and "矩形" in desc and "线段" in desc, f"描述应含图元摘要，实际 {desc}"

    def test_reexamine_includes_geometry(self):
        """重看增强：view_image 返回自动附主色 + 图元几何。"""
        from app.proxy import handle_chat_completion
        from app.config import get_config
        from app.image_cache import get_image_cache
        _clear = get_image_cache()
        asyncio.run(_clear.clear())
        cfg = get_config()
        old_r = cfg.image.vision_reexamine
        cfg.image.vision_reexamine = True
        try:
            raw = self._make_shapes_img()
            b64 = base64.b64encode(raw).decode()
            request = ChatCompletionRequest(
                model="openai-vision",
                messages=[ChatMessage(role="user", content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "这张图的形状?"},
                ])],
                stream=False,
            )
            class _ToolSource:
                def __init__(self):
                    self.bodies = []
                    self.calls = 0

                async def post(self, url, json=None):
                    self.bodies.append(json)
                    self.calls += 1
                    if self.calls == 1:
                        return _FakeResp2(200, {"choices": [{"message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [{"id": "c1", "type": "function", "function": {
                                "name": "grassvision_view_image",
                                "arguments": '{"question":"形状","region":"0,0,1000,1000"}'}}],
                        }}], "usage": {}})
                    return _FakeResp2(200, {"choices": [{"message": {"role": "assistant", "content": "有圆和矩形。"}}], "usage": {}})

                async def aclose(self):
                    pass

            source = _ToolSource()
            vision = _FakeVisionClient2()
            with patch("app.vision.get_vision_client", return_value=vision), \
                 patch("app.proxy.get_source_client", return_value=source):
                resp = asyncio.run(handle_chat_completion(request, type("R", (), {})()))
            import json as _json
            data = _json.loads(resp.body)
            assert data["choices"][0]["message"]["content"] == "有圆和矩形。"
            tool_msgs = [m for m in source.bodies[1]["messages"] if m.get("role") == "tool"]
            content = tool_msgs[0]["content"] if tool_msgs else ""
            assert "主色" in content, f"重看应附主色，实际 {content[:100]}"
            assert "图元" in content and "圆形" in content, f"重看应附图元几何，实际 {content[:150]}"
        finally:
            cfg.image.vision_reexamine = old_r
            asyncio.run(_clear.clear())
