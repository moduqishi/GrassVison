"""Anthropic Messages API 适配（POST /v1/messages）。

把 Anthropic 请求转换为内部 OpenAI 格式，复用核心管线；响应转回 Anthropic
格式（非流式 JSON 或流式 SSE events）。

图片：Anthropic content block image.source(base64/url) → OpenAI image_url(data URL)。
工具：Anthropic tools → OpenAI function tools；模型对客户端工具的调用（tool_use）
在内部为 OpenAI tool_calls，服务端无感的 grassvision 工具在内部闭环，
客户端只见最终回答；客户端自己的工具调用透传（转回 tool_use blocks）。
"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import HTTPException, Request

from app.schemas import ChatCompletionRequest, ChatMessage


# ───────────────────────── 入站解析 ─────────────────────────

def _extract_image_url(source: dict) -> str | None:
    """Anthropic image source → OpenAI data URL / http URL。"""
    stype = source.get("type")
    if stype == "base64":
        media = source.get("media_type", "image/png")
        data = source.get("data", "")
        return f"data:{media};base64,{data}"
    if stype == "url":
        return source.get("url")
    return None


def _block_to_content(block: dict) -> list[dict] | str | None:
    """单个 content block → OpenAI content 片段。"""
    btype = block.get("type")
    if btype == "text":
        return {"type": "text", "text": block.get("text", "")}
    if btype == "image":
        url = _extract_image_url(block.get("source") or {})
        if url:
            return {"type": "image_url", "image_url": {"url": url}}
        return None
    if btype == "tool_use":
        return {"__tool_use__": {
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "input": block.get("input") or {},
        }}
    if btype == "tool_result":
        return {"__tool_result__": {
            "tool_use_id": block.get("tool_use_id", ""),
            "content": block.get("content", ""),
        }}
    # thinking / 其他块：忽略（不传给源模型，避免格式污染）
    return None


def parse_messages_request(body: dict) -> ChatCompletionRequest:
    """Anthropic /v1/messages 请求 → 内部 ChatCompletionRequest（OpenAI 域）。"""
    raw_messages: list[dict] = []
    # system（Anthropic 独立字段）
    system = body.get("system")
    if system:
        if isinstance(system, str):
            raw_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            texts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                raw_messages.append({"role": "system", "content": "\n".join(t for t in texts if t)})

    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            if role == "assistant":
                raw_messages.append({"role": "assistant", "content": content})
            else:
                raw_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raw_messages.append({"role": role, "content": ""})
            continue
        # blocks 解析：按 role 组装 OpenAI 消息
        if role == "assistant":
            # 可能有 text 和 tool_use 混合
            texts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
            tool_uses = [b for b in content
                         if isinstance(b, dict) and b.get("type") == "tool_use"]
            if tool_uses:
                raw_messages.append({
                    "role": "assistant",
                    "content": "".join(texts) or None,
                    "tool_calls": [
                        {
                            "id": tu.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {"name": tu.get("name", ""),
                                         "arguments": json.dumps(tu.get("input") or {}, ensure_ascii=False)},
                        }
                        for i, tu in enumerate(tool_uses)
                    ],
                })
            else:
                raw_messages.append({"role": "assistant", "content": "".join(texts) or ""})
            continue
        if role == "user":
            parts: list[dict] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    raw_messages.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id", ""),
                        "content": b.get("content", ""),
                    })
                else:
                    p = _block_to_content(b)
                    if isinstance(p, dict):
                        parts.append(p)
            if parts:
                raw_messages.append({"role": "user", "content": parts})
            continue
        # 其他 role（model 等）：忽略
        continue

    # 工具转换：Anthropic tools → OpenAI function tools
    openai_tools: list[dict] = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })

    return ChatCompletionRequest(
        model=body.get("model", ""),
        messages=[ChatMessage(**m) for m in raw_messages],
        stream=bool(body.get("stream", False)),
        tools=openai_tools if openai_tools else None,
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        top_p=body.get("top_p"),
    )


# ───────────────────────── 出站序列化 ─────────────────────────

def _usage_to_anthropic(usage: dict) -> dict:
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


def build_messages_response(data: dict, model: str) -> dict:
    """内部 OpenAI 非流式响应 → Anthropic message JSON。"""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content_blocks: list[dict] = []
    stop_reason = "end_turn"
    text = msg.get("content") or ""
    if text:
        content_blocks.append({"type": "text", "text": text})
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        stop_reason = "tool_use"
        for tc in tool_calls:
            fn = tc.get("function") or {}
            try:
                inp = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                inp = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": fn.get("name", ""),
                "input": inp,
            })
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _usage_to_anthropic(data.get("usage") or {}),
    }


def iter_anthropic_stream(raw_request: Request, model: str, openai_stream) -> None:
    """把内部 OpenAI 流式响应（SSE 帧）转换为 Anthropic 流式 events。

    规则：
      - OpenAI content 增量 → content_block_delta(text_delta)
      - OpenAI reasoning_content 增量 → 忽略（Anthropic thinking 需要 signature，暂不输出）
      - OpenAI tool_calls（客户端工具透传）→ content_block_start(tool_use) 一次性输出
      - finish_reason / [DONE] → content_block_stop + message_delta + message_stop
    """
    import asyncio

    # 返回 async generator
    async def _gen():
        message_id = f"msg_{uuid.uuid4().hex}"
        started = False       # message_start + 首个 content_block_start(text) 已发
        tool_block_index = 0  # 已发出的 content block 数
        text_index = None
        stop_reason = "end_turn"

        def _evt(name: str, payload: dict) -> str:
            return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield _evt("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id, "type": "message", "role": "assistant",
                "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        async for raw_line in openai_stream:
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = ((chunk.get("choices") or [{}])[0].get("delta")) or {}
            finish = ((chunk.get("choices") or [{}])[0].get("finish_reason"))
            content = delta.get("content")
            tcs = delta.get("tool_calls")
            if content and not started:
                # 首个文本增量：content_block_start(text) + delta
                yield _evt("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                started = True
                text_index = 0
            if content:
                yield _evt("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_index or 0,
                    "delta": {"type": "text_delta", "text": content},
                })
            if tcs:
                # 客户端工具调用透传：输出 tool_use block
                for tc in tcs:
                    fn = tc.get("function") or {}
                    if not fn.get("name"):
                        continue
                    try:
                        inp = json.loads(fn.get("arguments") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        inp = {}
                    idx = (text_index + 1) if text_index is not None else (tool_block_index + 1)
                    yield _evt("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": fn.get("name", ""),
                            "input": inp,
                        },
                    })
                    yield _evt("content_block_stop", {
                        "type": "content_block_stop", "index": idx,
                    })
                    tool_block_index = idx
                    stop_reason = "tool_use"
            if finish == "tool_calls":
                stop_reason = "tool_use"
            elif finish == "stop":
                stop_reason = "end_turn"
        # 收尾
        if not started:
            yield _evt("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
            started = True
        yield _evt("content_block_stop", {
            "type": "content_block_stop", "index": text_index or 0,
        })
        yield _evt("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })
        yield _evt("message_stop", {"type": "message_stop"})

    return _gen()
