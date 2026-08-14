"""Core proxy: chat completions routing, streaming, and vision enhancement."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import get_config
from app.errors import ModelNotFoundError, VisionAnalysisError, ProviderError
from app.image_utils import (
    ExtractedImage,
    extract_all_images_with_positions,
    extract_current_turn_positions,
    inject_image_descriptions,
    assert_no_image_url_blocks,
    extract_user_question,
)
from app.providers import get_source_client
from app.schemas import ChatCompletionRequest, EnhancedModelConfig
from app.stats import get_stats
from app.vision import resolve_image_descriptions, _merge_and_number_descriptions, _build_injection_text, reexamine_image

# 协议化服务端重看：注入给源模型的工具定义。模型描述不足时自主调用，
# GrassVision 在服务端执行（用请求内图片字节 + 新意图重新分析），客户端无感知。
_VIEW_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "grassvision_view_image",
        "description": (
            "重新分析图片获取准确信息。图片包括：用户当前上传的图片，以及对话历史中用户发送过的图片。"
            "当已有图片分析（<grassvision_image_context>）中缺少回答用户问题所需的细节"
            "（颜色、坐标、小字、图标、布局、表格内容等）时，必须调用本工具重新查看图片，"
            "不要猜测或假设分析已完整。返回针对你问题的最新分析结果。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "你想针对图片问的具体问题或需要查看的细节"},
                "region": {"type": "string", "description": "可选，0-1000 归一化区域 x1,y1,x2,y2（逗号分隔），只查看该区域"},
            },
            "required": ["question"],
        },
    },
}

# 本地确定性像素工具（服务端执行，不依赖视觉模型，数值精确）
_PIXEL_COLORS_TOOL = {
    "type": "function",
    "function": {
        "name": "grassvision_pixel_colors",
        "description": (
            "精确获取图片区域的主色调（本地像素算法，返回 #RRGGBB 色值与占比）。"
            "当需要准确的色值（如按钮/背景/图表的精确颜色）而不是大致描述时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "可选，0-1000 归一化区域 x1,y1,x2,y2"},
                "candidates": {"type": "array", "items": {"type": "string"},
                               "description": "可选，候选色值列表（如 #F3F4F6），返回其中最接近的"},
            },
        },
    },
}

_PIXEL_DIFF_TOOL = {
    "type": "function",
    "function": {
        "name": "grassvision_pixel_diff",
        "description": (
            "精确对比图片中两个区域的像素差异（差异百分比 + 最差区域坐标）。"
            "用于判断两处是否一致、找不同、验证改版对齐。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region_a": {"type": "string", "description": "0-1000 归一化区域 x1,y1,x2,y2"},
                "region_b": {"type": "string", "description": "0-1000 归一化区域 x1,y1,x2,y2"},
            },
            "required": ["region_a", "region_b"],
        },
    },
}

_PIXEL_TRACE_TOOL = {
    "type": "function",
    "function": {
        "name": "grassvision_trace",
        "description": (
            "精确提取图片区域的几何信息（前景包围盒像素坐标、宽高、占比、主色、边缘轨迹SVG）。"
            "当需要准确的尺寸/形状/几何关系时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "可选，0-1000 归一化区域 x1,y1,x2,y2"},
            },
        },
    },
}

_GRASSVISION_TOOL_NAMES = (
    "grassvision_view_image",
    "grassvision_pixel_colors",
    "grassvision_pixel_diff",
    "grassvision_trace",
)


def _is_grassvision_tool(name: str) -> bool:
    return name in _GRASSVISION_TOOL_NAMES


def _inject_grassvision_tools(body: dict, reexamine: bool, pixel_tools: bool) -> dict:
    """注入 GrassVision 服务端工具（按开关）：view_image（重看）+ 像素工具。"""
    tools = list(body.get("tools") or [])
    existing = {
        t.get("function", {}).get("name") for t in tools
        if isinstance(t, dict) and t.get("type") == "function"
    }
    if reexamine and "grassvision_view_image" not in existing:
        tools.append(_VIEW_IMAGE_TOOL)
    if pixel_tools:
        for tool in (_PIXEL_COLORS_TOOL, _PIXEL_DIFF_TOOL, _PIXEL_TRACE_TOOL):
            if tool["function"]["name"] not in existing:
                tools.append(tool)
    return {**body, "tools": tools}


def _strip_grassvision_tools(body: dict) -> dict:
    """移除所有 grassvision_* 工具（客户端原有工具保留）；无剩余工具时去掉 tools 键。"""
    tools = [
        t for t in (body.get("tools") or [])
        if not (isinstance(t, dict)
                and _is_grassvision_tool(t.get("function", {}).get("name", "")))
    ]
    if not tools:
        return {k: v for k, v in body.items() if k != "tools"}
    return {**body, "tools": tools}

# 服务端重看最大轮数（一次请求内模型最多自主重看几次）
MAX_REEXAMINE_ROUNDS = 3


def _collect_tool_calls(tool_deltas: list[dict]) -> list[dict]:
    """把流式 tool_calls 增量（按 index 累计）组装成完整 tool_call 列表。"""
    acc: dict[int, dict] = {}
    for tc in tool_deltas:
        if not isinstance(tc, dict):
            continue
        idx = tc.get("index", 0)
        entry = acc.setdefault(idx, {"id": None, "name": None, "args": ""})
        if tc.get("id"):
            entry["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            entry["name"] = fn["name"]
        if fn.get("arguments"):
            entry["args"] += fn["arguments"]
    out = []
    for idx in sorted(acc):
        e = acc[idx]
        out.append({
            "id": e["id"] or f"call_{idx}",
            "type": "function",
            "function": {"name": e["name"] or "", "arguments": e["args"]},
        })
    return out


def _assistant_msg_with_tool_calls(tool_calls: list[dict]) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _accum_usage(target: dict, usage: dict | None) -> None:
    """把一次调用的 usage（prompt/completion/total 等）累加到 target。"""
    for k, v in (usage or {}).items():
        if isinstance(v, (int, float)):
            target[k] = target.get(k, 0) + v


async def _resolve_image_raw(url: str, request_client) -> bytes | None:
    """从图片 URL（data URL 或 http）解析原始字节，供本地像素工具使用。"""
    import base64 as _b64
    try:
        from app.image_utils import DATA_URL_RE as _DURL
        m = _DURL.match(url)
        if m:
            return _b64.b64decode(m.group(2))
        from app.image_utils import fetch_image_bytes
        return await fetch_image_bytes(url, request_client)
    except Exception:
        return None


async def _execute_grassvision_tool(
    tc: dict,
    images: list,
    model: EnhancedModelConfig,
    raw_request: Request,
    usage_accumulator: dict | None = None,
) -> str:
    """服务端执行 grassvision_* 工具（重看 / 本地像素工具），返回给源模型的 tool 结果文本。"""
    from app import pixel_tools as PT

    name = tc.get("function", {}).get("name", "")
    try:
        args = json.loads(tc.get("function", {}).get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    if not images:
        return "[当前没有可重看的图片]"
    url = images[0].url

    if name == "grassvision_view_image":
        return await reexamine_image(
            url, str(args.get("question", "") or ""), args.get("region"),
            model, getattr(raw_request, "_httpx_client", None),
            usage_accumulator=usage_accumulator,
        )

    # 本地像素工具（确定性算法）
    raw = await _resolve_image_raw(url, getattr(raw_request, "_httpx_client", None))
    if raw is None:
        return "[图片解析失败]"

    if name == "grassvision_pixel_colors":
        colors = PT.dominant_colors(
            raw, region=args.get("region"),
            candidates=args.get("candidates"),
        )
        lines = []
        for c in colors:
            share = c.get("share")
            if c.get("matched"):
                lines.append(f"- {c['color']}（候选匹配）")
            else:
                lines.append(f"- {c['color']} 占比 {share * 100:.1f}%")
        return "区域主色调：\n" + "\n".join(lines) if lines else "[未提取到颜色]"

    if name == "grassvision_pixel_diff":
        result = PT.pixel_diff(raw, str(args.get("region_a", "")), str(args.get("region_b", "")))
        if "error" in result:
            return f"[{result['error']}]"
        box = result.get("worst_region")
        box_txt = f"（0-1000 归一化 {box}）" if box else ""
        return (
            f"两区域像素差异：{result['diff_percent']}%"
            f"，平均通道差 {result['mean_diff']}"
            f"，最差子区域差异 {result['worst_share']}% {box_txt}"
        )

    if name == "grassvision_trace":
        geo = PT.trace_region(raw, region=args.get("region"))
        if "error" in geo:
            return f"[{geo['error']}]"
        box = geo.get("foreground_box_px")
        return (
            f"前景几何（原图像素坐标）：包围盒 {box}，宽 {geo.get('width')}px 高 {geo.get('height')}px，"
            f"前景占比 {geo.get('coverage')}，主色 {geo.get('dominant_color')}\n"
            f"边缘轨迹 SVG（原图坐标）：\n{geo.get('svg')}"
        )

    return f"[未知工具 {name}]"


async def _forward_with_reexamine(
    body: dict,
    provider_key: str,
    public_model_id: str,
    model: EnhancedModelConfig,
    images: list,
    raw_request: Request,
) -> tuple[JSONResponse, dict | None, dict]:
    """非流式转发 + 服务端工具重看 loop。

    拦截源模型的 grassvision_view_image 调用 → 服务端执行重看（重新调视觉模型）
    → 把 tool result 喂回源模型再转发 → 直到源模型给出最终回答（或达到轮数上限）。
    客户端只看到最终回答，中间的"自主多次看图"完全在服务端完成。

    返回 (resp, agg_source_usage, agg_vision_usage)：用量为多轮重看的聚合值，
    供上层一次性记录，避免重复计数或丢失。
    """
    agg_source: dict = {}
    agg_vision: dict = {}
    body_plain: dict | None = None
    for _round in range(MAX_REEXAMINE_ROUNDS + 1):
        resp, usage = await _forward_to_source(
            body, provider_key, public_model_id, stream=False,
        )
        _accum_usage(agg_source, usage)
        # 上游可能不支持 tools 字段（400/422）→ 自动回退为无工具请求重试一次
        if resp.status_code in (400, 422) and body_plain is None and body.get("tools"):
            body_plain = _strip_grassvision_tools(body)
            body = body_plain
            resp, usage = await _forward_to_source(
                body, provider_key, public_model_id, stream=False,
            )
            _accum_usage(agg_source, usage)
        try:
            data = json.loads(resp.body)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return resp, agg_source, agg_vision
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        gv_calls = [
            tc for tc in tool_calls
            if isinstance(tc, dict) and _is_grassvision_tool(tc.get("function", {}).get("name", ""))
        ]
        if not gv_calls:
            return resp, agg_source, agg_vision

        if _round >= MAX_REEXAMINE_ROUNDS:
            # 达到重看上限：把 assistant 消息与工具说明追加进对话，
            # 移除工具定义，强制源模型基于已有图片信息直接回答（不泄漏 tool_calls）
            body["messages"] = list(body.get("messages") or []) + [msg]
            body["messages"].append({
                "role": "tool",
                "tool_call_id": (gv_calls[0].get("id") if gv_calls else "call_none"),
                "content": "[已达到工具调用次数上限，无法继续；请基于已有的图片分析信息直接回答用户。]",
            })
            body = _strip_grassvision_tools(body)
            resp, usage = await _forward_to_source(
                body, provider_key, public_model_id, stream=False,
            )
            _accum_usage(agg_source, usage)
            return resp, agg_source, agg_vision

        # 服务端执行 grassvision_* 工具：把 assistant 消息（含 tool_calls）与
        # 工具结果追加进对话再转发（客户端无感知）
        body["messages"] = list(body.get("messages") or []) + [msg]
        for tc in gv_calls:
            try:
                result_text = await _execute_grassvision_tool(
                    tc, images, model, raw_request, usage_accumulator=agg_vision,
                )
            except Exception as e:
                result_text = f"[工具执行失败: {e}]"
            body["messages"].append({
                "role": "tool", "tool_call_id": tc.get("id"), "content": result_text,
            })
    return resp, agg_source, agg_vision


async def _stream_with_reexamine(
    body: dict,
    provider_key: str,
    public_model_id: str,
    model: EnhancedModelConfig,
    images: list,
    raw_request: Request,
    on_usage: callable | None = None,
    extra_usage: dict | None = None,
    stream_vision: bool = False,
    stream_id: str = "",
    is_first: dict | None = None,
):
    """流式转发 + 服务端重看（单条 SSE 流，客户端无感知）。

    逐帧转发源模型流；首次检测到 tool_calls 增量时按工具名分流：
      - grassvision_view_image → 该轮流整体吞掉（客户端看不到），流结束后
        服务端执行重看并发起下一轮流；
      - 客户端自己的其他工具 → 原样转发（不影响 harness 自带工具）。
    达到重看上限时移除工具并强制源模型直接回答。
    stream_vision=True 时，重看的视觉增量以 reasoning_content 帧实时推给客户端
    （融合版：视觉思考① → 源模型思考 → 视觉思考② → 源模型回答）。

    用量统计：多轮重看的源模型/视觉 token 聚合后统一上报一次（on_usage），
    避免重复计数或丢失。
    """
    cfg = get_config()
    if is_first is None:
        is_first = {"value": True}
    if not stream_id:
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    agg_source: dict = {}
    agg_vision: dict = {}

    async def _inner():
        nonlocal body
        plain_retried = False
        for _round in range(MAX_REEXAMINE_ROUNDS + 2):
            provider = cfg.source_providers.get(provider_key)
            if not provider or not provider.enabled:
                yield f"data: {json.dumps({'error': {'message': 'Source provider unavailable'}})}" + "\n\n"
                yield "data: [DONE]\n\n"
                return
            client = get_source_client(provider)
            mode = "forward"      # forward=转发给客户端 | swallow=吞掉工具轮
            tool_deltas: list[dict] = []
            finished = False
            try:
                async with client.stream("POST", "/chat/completions", json=body) as resp:
                    if resp.status_code != 200:
                        error_body = (await resp.aread()).decode(errors="replace")[:500]
                        # 上游不支持 tools → 回退无工具重试一次
                        if not plain_retried and body.get("tools"):
                            plain_retried = True
                            body = _strip_grassvision_tools(body)
                            continue
                        err_frame = json.dumps({'error': {'message': error_body, 'type': 'upstream_error', 'code': resp.status_code}})
                        yield f"data: {err_frame}" + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            if mode == "forward":
                                yield "\n"
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                if mode == "forward":
                                    yield "data: [DONE]\n\n"
                                finished = True
                                break
                            u = _extract_usage_from_chunk(data_str)
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            delta = ((data.get("choices") or [{}])[0].get("delta")) or {}
                            tcs = delta.get("tool_calls")
                            if tcs:
                                if mode == "forward":
                                    # 首帧工具调用：按工具名分流
                                    names = [
                                        (tc.get("function") or {}).get("name") or ""
                                        for tc in tcs if isinstance(tc, dict)
                                    ]
                                    if any(_is_grassvision_tool(n) for n in names):
                                        mode = "swallow"
                                    # 客户端自己的工具：保持 forward 正常转发
                                if mode == "swallow":
                                    tool_deltas.extend(tcs)
                                    continue
                            if mode == "forward":
                                # 聚合用量（不逐帧 record，generator 结束时统一上报一次）
                                _accum_usage(agg_source, u)
                                if u and extra_usage:
                                    try:
                                        data["usage"] = {**u, **extra_usage}
                                        line = f"data: {json.dumps(data, ensure_ascii=False)}"
                                    except json.JSONDecodeError:
                                        pass
                                if is_first["value"]:
                                    line = _sanitize_stream_chunk(line, public_model_id, True)
                                    is_first["value"] = False
                                else:
                                    line = _sanitize_stream_chunk(line, public_model_id, False)
                                yield line + "\n"
                        elif mode == "forward" and line.strip():
                            yield line + "\n"
            finally:
                pass  # 连接池化，不在调用点关闭

            if not finished and mode == "forward":
                # 流异常中断但已转发——结束
                return

            if mode == "forward":
                return  # 正常转发完成（含客户端自己的工具透传）

            # ── 吞掉了 grassvision_* 工具轮：服务端执行 ──
            tool_calls = _collect_tool_calls(tool_deltas)
            gv_calls = [
                tc for tc in tool_calls
                if _is_grassvision_tool(tc.get("function", {}).get("name", ""))
            ]
            if not gv_calls:
                return
            assistant_msg = _assistant_msg_with_tool_calls(tool_calls)
            body["messages"] = list(body.get("messages") or []) + [assistant_msg]

            if _round >= MAX_REEXAMINE_ROUNDS:
                # 达到上限：移除工具 + 说明，强制下一轮直接回答
                body = _strip_grassvision_tools(body)
                body["messages"].append({
                    "role": "tool",
                    "tool_call_id": gv_calls[0].get("id", "call_none"),
                    "content": "[已达到工具调用次数上限，无法继续；请基于已有的图片分析信息直接回答用户。]",
                })
                continue

            for tc in gv_calls:
                if not images:
                    body["messages"].append({
                        "role": "tool", "tool_call_id": tc.get("id"),
                        "content": "[当前没有可用的图片]",
                    })
                    continue
                try:
                    name = tc.get("function", {}).get("name", "")
                    if stream_vision and name == "grassvision_view_image":
                        # 融合版：重看的视觉增量实时推给客户端（reasoning_content 帧）
                        try:
                            args = json.loads(tc.get("function", {}).get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        question = str(args.get("question", "") or "")
                        region = args.get("region")
                        vq: asyncio.Queue = asyncio.Queue()

                        async def _vemit(kind: str, text: str) -> None:
                            await vq.put(("token", kind, text))

                        async def _vrun() -> str:
                            try:
                                return await reexamine_image(
                                    images[0].url, question, region, model,
                                    getattr(raw_request, "_httpx_client", None),
                                    emit=_vemit,
                                    usage_accumulator=agg_vision,
                                )
                            finally:
                                await vq.put(None)

                        vtask = asyncio.create_task(_vrun())
                        while True:
                            vitem = await vq.get()
                            if vitem is None:
                                break
                            vframe, is_first["value"] = _build_vision_frame(
                                vitem[2], public_model_id, stream_id, is_first["value"]
                            )
                            yield vframe
                        result_text = await vtask
                    else:
                        result_text = await _execute_grassvision_tool(
                            tc, images, model, raw_request, usage_accumulator=agg_vision,
                        )
                except Exception as e:
                    result_text = f"[工具执行失败: {e}]"
                body["messages"].append({
                    "role": "tool", "tool_call_id": tc.get("id"), "content": result_text,
                })

    try:
        async for frame in _inner():
            yield frame
    finally:
        # 用量聚合：一次请求（含多轮重看）只上报一次
        if on_usage and agg_source:
            on_usage(agg_source)



def _find_model(model_id: str) -> EnhancedModelConfig:
    cfg = get_config()
    model = cfg.models.get(model_id)
    if not model or not model.enabled:
        raise ModelNotFoundError(model_id)
    return model


# 思考链引导：系统设置 image.thinking_guidance 开启时注入系统提示，
# 要求源模型在推理过程中引用图片分析结果。
_THINKING_GUIDANCE_TEXT = (
    "用户消息的 <grassvision_image_context> 标签内附带了从用户上传图片中自动提取的分析信息。\n"
    "请先在你的思考过程（推理链）中仔细阅读并引用这些图片分析信息，"
    "结合图片内容完成推理后，再回答用户的问题。"
)

# 通道说明：image.vision_channel_note 开启时注入。告诉源模型"收到的是文字分析不是像素"，
# 细节不足时引导按需重看（模拟原生多模态，参考 agent-vision-toolkit 的通道说明设计）。
# 有工具版（vision_reexamine 开启）：引导调用 grassvision_view_image 工具重新分析；
# 无工具版（仅开通道说明）：引导用户重新发送图片（重发触发带新意图的重新分析）。
_CHANNEL_NOTE_TEXT = (
    "重要说明：<grassvision_image_context> 中的图片信息是视觉模型对用户图片的"
    "文字分析，不是图片本身——你看不到像素，而且分析**可能不完整或有遗漏**。\n"
    "如果回答用户问题需要分析中没有的细节（颜色、坐标、小字、图标、表格内容等），"
    "**不要猜测、不要假设分析已覆盖全部内容**；请调用 grassvision_view_image 工具"
    "重新分析图片获取准确信息。仅在无法重新分析时，再请用户重新发送图片。"
)

_CHANNEL_NOTE_TEXT_NO_TOOL = (
    "重要说明：<grassvision_image_context> 中的图片信息是视觉模型对用户图片的"
    "文字分析，不是图片本身——你看不到像素，而且分析**可能不完整或有遗漏**。\n"
    "如果回答用户问题需要分析中没有的细节（颜色、坐标、小字、图标、表格内容等），"
    "**不要猜测、不要假设分析已覆盖全部内容**；请明确指出缺失的具体细节，"
    "并请用户重新发送图片——重发后视觉模型会针对你的具体问题重新分析。"
)


def _inject_thinking_guidance(messages: list[dict]) -> list[dict]:
    """把思考链引导追加到已有的 system 消息；没有 system 消息则在最前面插入一条。"""
    guidance = _THINKING_GUIDANCE_TEXT
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                messages[i] = {**msg, "content": f"{content}\n\n{guidance}"}
            else:
                messages[i] = {"role": "system", "content": guidance}
            return messages
    return [{"role": "system", "content": guidance}, *messages]


def _inject_channel_note(messages: list[dict], with_tool: bool = True) -> list[dict]:
    """把通道说明追加到已有 system 消息；没有则插入一条（与思考链引导同理）。

    with_tool=True（vision_reexamine 开启）：引导调用重看工具；
    with_tool=False：引导用户重新发送图片。
    """
    note = _CHANNEL_NOTE_TEXT if with_tool else _CHANNEL_NOTE_TEXT_NO_TOOL
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                messages[i] = {**msg, "content": f"{content}\n\n{note}"}
            else:
                messages[i] = {"role": "system", "content": note}
            return messages
    return [{"role": "system", "content": note}, *messages]
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                messages[i] = {**msg, "content": f"{content}\n\n{_CHANNEL_NOTE_TEXT}"}
            else:
                messages[i] = {"role": "system", "content": _CHANNEL_NOTE_TEXT}
            return messages
    return [{"role": "system", "content": _CHANNEL_NOTE_TEXT}, *messages]


def _build_source_body(request: ChatCompletionRequest, model: EnhancedModelConfig, messages: list[dict]) -> dict:
    body = {
        "model": model.source_model,
        "messages": messages,
        "stream": request.stream,
    }
    for key in ("temperature", "top_p", "max_tokens", "tools", "tool_choice",
                "stop", "frequency_penalty", "presence_penalty", "seed", "n", "user"):
        val = getattr(request, key, None)
        if val is not None:
            body[key] = val
    return body


def _sanitize_stream_chunk(chunk_text: str, public_model_id: str, is_first: bool) -> str:
    if not is_first or not chunk_text.startswith("data: "):
        return chunk_text
    try:
        data_str = chunk_text[6:].strip()
        if data_str == "[DONE]":
            return chunk_text
        data = json.loads(data_str)
        data["model"] = public_model_id
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    except (json.JSONDecodeError, KeyError):
        return chunk_text


def _extract_usage_from_chunk(data_str: str) -> dict | None:
    try:
        if not data_str or data_str == "[DONE]":
            return None
        data = json.loads(data_str)
        return data.get("usage")
    except (json.JSONDecodeError, TypeError):
        return None


def _vision_usage_extra(vision_usage: dict | None) -> dict | None:
    """把视觉模型 token 用量转成 vision_* 前缀字段，并入响应 usage（不覆盖标准字段）。"""
    if not vision_usage:
        return None
    extra = {f"vision_{k}": v for k, v in vision_usage.items() if isinstance(v, (int, float))}
    return extra or None


def _openai_error_response(status: int, message: str, error_type: str = "grassvision_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "code": status}},
    )


def _build_vision_frame(
    text: str,
    public_model_id: str,
    stream_id: str,
    is_first: bool,
) -> tuple[str, bool]:
    """把视觉模型的分析增量包装成 SSE chunk，统一放入 delta.reasoning_content，
    使客户端把它们显示为思考链的一部分（与源模型的思考链无缝衔接）。"""
    delta: dict = {}
    if is_first:
        delta["role"] = "assistant"
    delta["reasoning_content"] = text
    data = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": public_model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n", False


async def handle_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
) -> JSONResponse | StreamingResponse:
    cfg = get_config()
    model = _find_model(request.model)
    stats_tracker = get_stats()

    messages_raw = [m.model_dump(exclude_none=True) for m in request.messages]

    # ── 1. Extract all images with positions ─────────────────────
    all_images = extract_all_images_with_positions(messages_raw)

    # ── 1.5 只处理「当前轮次」里的图片 ────────────────────────
    # 当前轮次 = 最后一次 assistant 回复之后的用户消息（兼容图片/文字分两条消息发）。
    # 更早轮次的图片一律剥离：不分析、不注入、不参与流式思考。
    # 解决：无关图片污染上下文、模型反复提及旧图、纯文字追问被拖慢首 token。
    current_positions = extract_current_turn_positions(messages_raw)
    current_images = [img for img in all_images if img.position in current_positions]

    if not current_images or not model.vision_enabled:
        # 无当前图片或视觉关闭 → 剥离所有图片块后直接转发
        stripped = messages_raw
        if all_images:
            if model.vision_enabled and cfg.image.reuse_historical_cache and not current_images:
                # 历史图片缓存复用：纯文字追问上一轮图片时，用缓存描述原地注入
                # （不触发新分析；未命中按 historical_cache_miss 处理）。
                try:
                    descriptions, _ = await resolve_image_descriptions(
                        images=all_images,
                        model_config=model,
                        allow_analysis_positions=set(),
                        historical_cache_miss=cfg.image.historical_cache_miss,
                        request_client=getattr(raw_request, "_httpx_client", None),
                        user_question=extract_user_question(messages_raw),
                        failure_mode=model.vision_failure_mode,
                    )
                    stripped = inject_image_descriptions(messages_raw, descriptions)
                except VisionAnalysisError:
                    stripped = inject_image_descriptions(messages_raw, {})
            else:
                stripped = inject_image_descriptions(messages_raw, {})
        # 无当前图片但开启重看：注入通道说明（引导模型怀疑描述、主动重看历史图）
        if model.vision_enabled and (cfg.image.vision_reexamine or cfg.image.pixel_tools) and cfg.image.vision_channel_note:
            stripped = _inject_channel_note(stripped)
        body = _build_source_body(request, model, stripped)
        if request.stream:
            def _on_noimg_usage(usage):
                stats_tracker.record_call(
                    model=request.model, images=len(all_images),
                    stream=True, elapsed=0,
                    vision_used=False, vision_success=False,
                    source_tokens=usage,
                )
            if model.vision_enabled and (cfg.image.vision_reexamine or cfg.image.pixel_tools):
                # 无图流式：注入工具，允许源模型重看历史图片（跨轮次无感重看）。
                # stream_vision=True：跨轮重看的思考链也始终流式透传。
                body = _inject_grassvision_tools(body, cfg.image.vision_reexamine, cfg.image.pixel_tools)
                return StreamingResponse(
                    _stream_with_reexamine(
                        body=body,
                        provider_key=model.source_provider,
                        public_model_id=request.model if model.replace_response_model else model.source_model,
                        model=model,
                        images=all_images,
                        raw_request=raw_request,
                        on_usage=_on_noimg_usage,
                        stream_vision=True,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            resp, _ = await _forward_to_source(
                body=body,
                provider_key=model.source_provider,
                public_model_id=request.model if model.replace_response_model else model.source_model,
                stream=True,
                on_usage=_on_noimg_usage,
            )
        else:
            noimg_vision_usage: dict | None = None
            if model.vision_enabled and (cfg.image.vision_reexamine or cfg.image.pixel_tools):
                # 无图非流式：注入工具，源模型可重看历史图片（跨轮次无感重看）
                body = _inject_grassvision_tools(body, cfg.image.vision_reexamine, cfg.image.pixel_tools)
                resp, source_usage, reexam_vision = await _forward_with_reexamine(
                    body=body,
                    provider_key=model.source_provider,
                    public_model_id=request.model if model.replace_response_model else model.source_model,
                    model=model,
                    images=all_images,
                    raw_request=raw_request,
                )
                if reexam_vision:
                    noimg_vision_usage = reexam_vision
            else:
                resp, source_usage = await _forward_to_source(
                    body=body,
                    provider_key=model.source_provider,
                    public_model_id=request.model if model.replace_response_model else model.source_model,
                    stream=False,
                )
            stats_tracker.record_call(
                model=request.model, images=len(all_images),
                stream=request.stream, elapsed=0,
                vision_used=bool(noimg_vision_usage), vision_success=bool(noimg_vision_usage),
                vision_tokens=noimg_vision_usage,
                source_tokens=source_usage,
            )
        return resp

    # Validate image count（仅当前图片）
    if len(current_images) > cfg.image.max_images:
        return _openai_error_response(400, f"Too many images: {len(current_images)} > {cfg.image.max_images}")

    # ── 2.5 融合流式：视觉思考（可选）+ 源模型流（含服务端重看）──
    # 统一入口：stream_vision_thinking 或 vision_reexamine 任一开启即走融合流，
    # 二者可自由组合（不再二选一）。
    if request.stream and (cfg.image.stream_vision_thinking or cfg.image.vision_reexamine or cfg.image.pixel_tools):
        return StreamingResponse(
            _combined_stream(
                request=request,
                model=model,
                messages_raw=messages_raw,
                current_images=current_images,
                current_positions=current_positions,
                raw_request=raw_request,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # ── 3. Resolve 图片描述（缓存 + 视觉调用）────────────
    # reuse_historical_cache 开启时把历史图片一并传入：缓存命中的历史描述原地
    # 注入（保住多轮追问上下文），未命中按 historical_cache_miss 处理，不新增分析。
    resolve_images = all_images if cfg.image.reuse_historical_cache else current_images
    try:
        descriptions, vision_usage = await resolve_image_descriptions(
            images=resolve_images,
            model_config=model,
            allow_analysis_positions=current_positions,
            historical_cache_miss=cfg.image.historical_cache_miss,
            request_client=getattr(raw_request, "_httpx_client", None),
            user_question=extract_user_question(messages_raw),
            failure_mode=model.vision_failure_mode,
        )
    except VisionAnalysisError as e:
        if model.vision_failure_mode == "skip":
            # fail-open（借鉴 agent-vision-toolkit）：剥离图片 + 注入失败说明，
            # 而不是把 image_url 原样发给纯文本源模型导致上游报错
            note = f"[图片分析失败：{e.message}，已跳过视觉分析]"
            replacements = {img.position: note for img in current_images}
            stripped = inject_image_descriptions(messages_raw, replacements)
            body = _build_source_body(request, model, stripped)
            if request.stream:
                def _on_skip_usage(usage):
                    stats_tracker.record_call(
                        model=request.model, images=len(all_images),
                        stream=True, elapsed=0,
                        vision_used=True, vision_success=False,
                        vision_tokens=None,
                        source_tokens=usage,
                    )
                resp, _ = await _forward_to_source(
                    body=body,
                    provider_key=model.source_provider,
                    public_model_id=request.model if model.replace_response_model else model.source_model,
                    stream=True,
                    on_usage=_on_skip_usage,
                )
            else:
                resp, source_usage = await _forward_to_source(
                    body=body,
                    provider_key=model.source_provider,
                    public_model_id=request.model if model.replace_response_model else model.source_model,
                    stream=False,
                )
                stats_tracker.record_call(
                    model=request.model, images=len(all_images),
                    stream=request.stream, elapsed=0,
                    vision_used=True, vision_success=False,
                    vision_tokens=None,
                    source_tokens=source_usage,
                )
            return resp
        return _openai_error_response(502, f"Vision analysis failed: {e.message}")

    # ── 4. Inject descriptions into messages ─────────────────────
    enhanced_messages = inject_image_descriptions(messages_raw, descriptions)

    # ── 5. Merge and inject vision context into last user msg ────
    # 只把「当前图片」的描述合并进当前问题（历史图片的描述已原地注入，
    # 不再进入当前问题，避免旧图信息污染本轮提问）。
    current_descriptions = {
        pos: desc for pos, desc in descriptions.items() if pos in current_positions
    }
    merged = _merge_and_number_descriptions(current_descriptions)
    if merged:
        injection = _build_injection_text(merged)

        # Append injection to the last user message
        for i in range(len(enhanced_messages) - 1, -1, -1):
            if enhanced_messages[i].get("role") == "user":
                content = enhanced_messages[i].get("content")
                if isinstance(content, str):
                    enhanced_messages[i]["content"] = f"{content}\n\n{injection}"
                elif isinstance(content, list):
                    enhanced_messages[i]["content"] = list(content) + [{"type": "text", "text": f"\n{injection}"}]
                break

    # ── 6. 思考链引导：让源模型在推理时引用图片分析 ──────────
    if cfg.image.thinking_guidance:
        enhanced_messages = _inject_thinking_guidance(enhanced_messages)
    if cfg.image.vision_channel_note:
        enhanced_messages = _inject_channel_note(
            enhanced_messages, with_tool=cfg.image.vision_reexamine)

    # ── 7. Assert no image_url blocks remain ────────────────────
    assert_no_image_url_blocks(enhanced_messages)

    # ── 8. Forward to source model ──────────────────────────────
    body = _build_source_body(request, model, enhanced_messages)

    if request.stream:
        # For streaming, source token usage is captured asynchronously via callback
        def _on_source_usage(usage):
            stats_tracker.record_call(
                model=request.model, images=len(all_images),
                stream=True, elapsed=0,
                vision_used=bool(vision_usage), vision_success=True,
                vision_tokens=vision_usage if vision_usage else None,
                source_tokens=usage,
            )

        resp, _ = await _forward_to_source(
            body=body,
            provider_key=model.source_provider,
            public_model_id=request.model if model.replace_response_model else model.source_model,
            stream=True,
            on_usage=_on_source_usage,
            extra_usage=_vision_usage_extra(vision_usage),
        )
    else:
        if cfg.image.vision_reexamine or cfg.image.pixel_tools:
            # 协议化服务端工具：注入工具 + 拦截 tool_call 在服务端执行（客户端无感知）
            body = _inject_grassvision_tools(body, cfg.image.vision_reexamine, cfg.image.pixel_tools)
            resp, source_usage, reexam_vision = await _forward_with_reexamine(
                body=body,
                provider_key=model.source_provider,
                public_model_id=request.model if model.replace_response_model else model.source_model,
                model=model,
                images=current_images or all_images,
                raw_request=raw_request,
            )
            # 用量聚合：首次视觉分析 + 各轮重看的视觉 token
            if reexam_vision:
                vision_usage = dict(vision_usage or {})
                for k, v in reexam_vision.items():
                    vision_usage[k] = vision_usage.get(k, 0) + v
        else:
            resp, source_usage = await _forward_to_source(
                body=body,
                provider_key=model.source_provider,
                public_model_id=request.model if model.replace_response_model else model.source_model,
                stream=False,
                extra_usage=_vision_usage_extra(vision_usage),
            )
        stats_tracker.record_call(
            model=request.model, images=len(all_images),
            stream=request.stream, elapsed=0,
            vision_used=bool(vision_usage), vision_success=True,
            vision_tokens=vision_usage if vision_usage else None,
            source_tokens=source_usage,
        )

    return resp


async def _iter_source_stream(
    body: dict,
    provider_key: str,
    public_model_id: str,
    on_usage: callable | None = None,
    extra_usage: dict | None = None,
):
    """流式转发源模型：逐行 yield SSE 内容；收到 usage chunk 时回调 on_usage。
    extra_usage（如 vision_* token 统计）并入转发给客户端的 usage 块。"""
    cfg = get_config()
    provider = cfg.source_providers.get(provider_key)
    if not provider:
        raise ProviderError(f"Source provider '{provider_key}' not found", provider=provider_key)
    if not provider.enabled:
        raise ProviderError(f"Source provider '{provider_key}' is disabled", provider=provider_key, status_code=503)

    client = get_source_client(provider)
    try:
        async with client.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                error_data = json.dumps({
                    "error": {
                        "message": error_body.decode(errors="replace")[:500],
                        "type": "upstream_error",
                        "code": resp.status_code,
                    }
                })
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
                return
            first_chunk = True
            async for line in resp.aiter_lines():
                if not line:
                    yield "\n"
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield "data: [DONE]\n\n"
                        continue
                    u = _extract_usage_from_chunk(data_str)
                    if u and on_usage:
                        on_usage(u)
                    # 把视觉模型 token 用量并入 usage 块（vision_* 前缀，不覆盖标准字段）
                    if u and extra_usage:
                        try:
                            data = json.loads(data_str)
                            data["usage"] = {**u, **extra_usage}
                            line = f"data: {json.dumps(data, ensure_ascii=False)}"
                        except json.JSONDecodeError:
                            pass
                    if first_chunk:
                        line = _sanitize_stream_chunk(line, public_model_id, True)
                        first_chunk = False
                    else:
                        line = _sanitize_stream_chunk(line, public_model_id, False)
                    yield line + "\n"
                elif line.strip():
                    yield line + "\n"
    except httpx.TimeoutException:
        yield f"data: {json.dumps({'error': {'message': 'Source model timeout'}})}\n\n"
        yield "data: [DONE]\n\n"
    # 连接池化：client 由 providers 池统一管理，不在调用点关闭


async def _combined_stream(
    request: ChatCompletionRequest,
    model: EnhancedModelConfig,
    messages_raw: list[dict],
    current_images: list,
    current_positions: set,
    raw_request: Request,
):
    """流式透传视觉模型的思考/分析过程，再无缝衔接源模型的思考与回答（单条 SSE 流）。

    阶段 1：视觉分析调用改为流式，视觉模型的 reasoning 与 content 增量都以
            reasoning_content 形式推给客户端——首帧就是真实思考链（默认无预提示，
            可配置 image.vision_stream_prelude 开启占位提示）。
    阶段 2：图片描述注入完成后，直接衔接源模型流式转发（思考链 + 回答）。
    """
    cfg = get_config()
    stats_tracker = get_stats()
    public_model_id = request.model if model.replace_response_model else model.source_model
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    queue: asyncio.Queue = asyncio.Queue()
    is_first = {"value": True}
    # 融合版：视觉思考流是否开启（stream_vision_thinking 控制阶段 1 的可视化；
    # 即使关闭，阶段 2 的重看仍可用——重看增量同样可流式推送）
    vision_stream_on = bool(cfg.image.stream_vision_thinking)
    stream_queue = queue if vision_stream_on else None

    # 历史缓存复用需要全部图片位置（含历史轮次）
    all_images = extract_all_images_with_positions(messages_raw)

    # 预提示（可选）：默认关闭——首帧即视觉模型真实输出，思考链完全真实；
    # 开启则先推一条"正在处理"占位，消除下载/排队期的静默等待。
    if vision_stream_on and cfg.image.vision_stream_prelude:
        prelude, is_first["value"] = _build_vision_frame(
            "【正在处理用户发送的图片…】", public_model_id, stream_id, True
        )
        yield prelude

    # ── 阶段 1：视觉分析（可选流式透传思考）────────────────────
    resolve_images = all_images if cfg.image.reuse_historical_cache else current_images

    async def _resolve():
        try:
            return await resolve_image_descriptions(
                images=resolve_images,
                model_config=model,
                allow_analysis_positions=current_positions,
                historical_cache_miss=cfg.image.historical_cache_miss,
                request_client=getattr(raw_request, "_httpx_client", None),
                user_question=extract_user_question(messages_raw),
                stream_queue=stream_queue,
                failure_mode=model.vision_failure_mode,
            )
        finally:
            # 保证消费端一定能等到结束标记（即使内部抛异常也不会挂起）
            if stream_queue is not None:
                await queue.put(("done", "", ""))

    try:
        task = asyncio.create_task(_resolve())
        if stream_queue is not None:
            while True:
                item = await queue.get()
                if item[0] == "done":
                    break
                # item = ("token", kind, text)：kind ∈ {"reasoning", "content"}
                # 统一进 reasoning_content（视觉阶段的分析即思考链），kind 保留供诊断
                frame, is_first["value"] = _build_vision_frame(
                    item[2], public_model_id, stream_id, is_first["value"]
                )
                yield frame
        descriptions, vision_usage = await task
    except VisionAnalysisError as e:
        if model.vision_failure_mode != "skip":
            error_data = json.dumps({
                "error": {"message": f"Vision analysis failed: {e.message}",
                          "type": "grassvision_error", "code": 502},
            })
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"
            return
        # skip：剥离图片 + 注入失败说明后转发源模型（fail-open）
        note = f"[图片分析失败：{e.message}，已跳过视觉分析]"
        replacements = {img.position: note for img in current_images}
        stripped = inject_image_descriptions(messages_raw, replacements)
        body = _build_source_body(request, model, stripped)
        def _on_skip_usage(usage):
            stats_tracker.record_call(
                model=request.model, images=len(current_images),
                stream=True, elapsed=0,
                vision_used=True, vision_success=False,
                vision_tokens=None, source_tokens=usage,
            )
        async for line in _iter_source_stream(body, model.source_provider, public_model_id, on_usage=_on_skip_usage):
            yield line
        return

    # ── 阶段 2：注入描述并衔接源模型流 ────────────────────────
    enhanced_messages = inject_image_descriptions(messages_raw, descriptions)
    current_descriptions = {
        pos: desc for pos, desc in descriptions.items() if pos in current_positions
    }
    merged = _merge_and_number_descriptions(current_descriptions)
    if merged:
        injection = _build_injection_text(merged)

        for i in range(len(enhanced_messages) - 1, -1, -1):
            if enhanced_messages[i].get("role") == "user":
                content = enhanced_messages[i].get("content")
                if isinstance(content, str):
                    enhanced_messages[i]["content"] = f"{content}\n\n{injection}"
                elif isinstance(content, list):
                    enhanced_messages[i]["content"] = list(content) + [{"type": "text", "text": f"\n{injection}"}]
                break

    if cfg.image.thinking_guidance:
        enhanced_messages = _inject_thinking_guidance(enhanced_messages)
    if cfg.image.vision_channel_note:
        enhanced_messages = _inject_channel_note(
            enhanced_messages, with_tool=cfg.image.vision_reexamine)
    assert_no_image_url_blocks(enhanced_messages)

    body = _build_source_body(request, model, enhanced_messages)

    def _on_source_usage(usage):
        stats_tracker.record_call(
            model=request.model, images=len(current_images),
            stream=True, elapsed=0,
            vision_used=bool(vision_usage), vision_success=True,
            vision_tokens=vision_usage if vision_usage else None,
            source_tokens=usage,
        )

    if cfg.image.vision_reexamine or cfg.image.pixel_tools:
        # 融合版阶段 2：源模型流 + 服务端重看（工具轮吞掉、重看增量流式推送）。
        # stream_vision 恒 True：重看是"源模型主动再看一眼"的过程，思考链始终透传，
        # 不依赖首次视觉思考开关（stream_vision_thinking 只控制阶段 1 的可视化）。
        body = _inject_grassvision_tools(body, cfg.image.vision_reexamine, cfg.image.pixel_tools)
        async for line in _stream_with_reexamine(
            body=body,
            provider_key=model.source_provider,
            public_model_id=public_model_id,
            model=model,
            images=current_images or all_images,
            raw_request=raw_request,
            on_usage=_on_source_usage,
            extra_usage=_vision_usage_extra(vision_usage),
            stream_vision=True,
            stream_id=stream_id,
            is_first=is_first,
        ):
            yield line
    else:
        async for line in _iter_source_stream(
            body, model.source_provider, public_model_id,
            on_usage=_on_source_usage, extra_usage=_vision_usage_extra(vision_usage),
        ):
            yield line


async def _forward_to_source(
    body: dict,
    provider_key: str,
    public_model_id: str,
    stream: bool = False,
    on_usage: callable | None = None,
    extra_usage: dict | None = None,
) -> tuple[JSONResponse | StreamingResponse, dict | None]:
    cfg = get_config()
    provider = cfg.source_providers.get(provider_key)
    if not provider:
        raise ProviderError(f"Source provider '{provider_key}' not found", provider=provider_key)
    if not provider.enabled:
        raise ProviderError(f"Source provider '{provider_key}' is disabled", provider=provider_key, status_code=503)

    client = get_source_client(provider)

    if not stream:
        try:
            resp = await client.post("/chat/completions", json=body)
            if resp.status_code != 200:
                return _openai_error_response(resp.status_code, resp.text[:500]), None
            data = resp.json()
            usage = data.get("usage")
            # 视觉 token 用量并入响应 usage（vision_* 前缀，不覆盖标准字段）
            if extra_usage and isinstance(usage, dict):
                data["usage"] = {**usage, **extra_usage}
            if "model" in data:
                data["model"] = public_model_id
            return JSONResponse(content=data), usage
        except httpx.TimeoutException:
            return _openai_error_response(504, "Source model timeout"), None
        # 连接池化：client 由 providers 池统一管理，不在调用点关闭

    async def _stream_with_tracking():
        async for line in _iter_source_stream(
            body, provider_key, public_model_id, on_usage=on_usage, extra_usage=extra_usage
        ):
            yield line

    return (
        StreamingResponse(
            _stream_with_tracking(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        ),
        None,
    )
