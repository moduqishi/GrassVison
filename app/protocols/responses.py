"""OpenAI Responses API 适配（POST /v1/responses，Codex / 新版 OpenAI 客户端）。

请求 input 数组 → 内部 OpenAI 消息（input_text/input_image/function_call/
function_call_output → text/image_url/tool_calls/tool）；响应转回 Responses
格式（非流式 JSON 或流式 events：response.created → output_text.delta →
response.completed）。

图片：input_image.image_url → OpenAI image_url(data URL)。
"""
from __future__ import annotations

import json
import time
import uuid

from app.schemas import ChatCompletionRequest, ChatMessage


# ───────────────────────── 入站解析 ─────────────────────────

def parse_responses_request(body: dict) -> ChatCompletionRequest:
    raw_messages: list[dict] = []
    for item in body.get("input") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        role = item.get("role")
        # function_call（assistant 的工具调用）
        if itype == "function_call":
            try:
                args = json.loads(item.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            raw_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": item.get("name", ""),
                                 "arguments": json.dumps(args, ensure_ascii=False)},
                }],
            })
            continue
        # function_call_output（工具结果）
        if itype == "function_call_output":
            raw_messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })
            continue
        # 常规消息（user / assistant / system / developer）
        content = item.get("content")
        if isinstance(content, str):
            raw_messages.append({"role": role or "user", "content": content})
            continue
        if isinstance(content, list):
            if role == "assistant":
                texts = [p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "output_text" and p.get("text")]
                raw_messages.append({"role": "assistant", "content": "".join(texts) or ""})
                continue
            parts: list[dict] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                ptype = p.get("type")
                if ptype in ("input_text", "output_text"):
                    if p.get("text"):
                        parts.append({"type": "text", "text": p.get("text")})
                elif ptype in ("input_image", "image_url"):
                    url = p.get("image_url") or (p.get("image") or {}).get("url")
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
            if parts:
                raw_messages.append({"role": role or "user", "content": parts})
            else:
                raw_messages.append({"role": role or "user", "content": ""})
            continue
        raw_messages.append({"role": role or "user", "content": ""})

    # 工具转换：Responses function tools → OpenAI function tools
    openai_tools: list[dict] = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            },
        })

    return ChatCompletionRequest(
        model=body.get("model", ""),
        messages=[ChatMessage(**m) for m in raw_messages],
        stream=bool(body.get("stream", False)),
        tools=openai_tools if openai_tools else None,
        temperature=body.get("temperature"),
        max_tokens=body.get("max_output_tokens") or body.get("max_tokens"),
    )


# ───────────────────────── 出站序列化 ─────────────────────────

def _usage_to_responses(usage: dict) -> dict:
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    return {"input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct}


def build_responses_response(data: dict, model: str) -> dict:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    output: list[dict] = []
    text = msg.get("content") or ""
    if text:
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        output.append({
            "id": tc.get("id") or f"fc_{uuid.uuid4().hex[:8]}",
            "type": "function_call",
            "status": "completed",
            "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "name": fn.get("name", ""),
            "arguments": json.dumps(args, ensure_ascii=False),
        })
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": _usage_to_responses(data.get("usage") or {}),
    }


def iter_responses_stream(model: str, openai_stream):
    """内部 OpenAI 流式响应 → Responses events（response.created → ... → response.completed）。"""

    async def _gen():
        resp_id = f"resp_{uuid.uuid4().hex}"
        item_id = f"msg_{uuid.uuid4().hex}"
        text_buf: list[str] = []
        created = False
        item_started = False

        def _evt(name: str, payload: dict) -> str:
            return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 创建事件
        yield _evt("response.created", {
            "type": "response.created",
            "response": {
                "id": resp_id, "object": "response", "created_at": int(time.time()),
                "status": "in_progress", "model": model, "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
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
            content = delta.get("content")
            tcs = delta.get("tool_calls")
            if content and not item_started:
                yield _evt("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"id": item_id, "type": "message", "role": "assistant",
                             "status": "in_progress", "content": []},
                })
                yield _evt("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": item_id, "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
                item_started = True
            if content:
                text_buf.append(content)
                yield _evt("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": item_id, "output_index": 0, "content_index": 0,
                    "delta": content,
                })
            if tcs:
                # 客户端工具调用透传
                for tc in tcs:
                    fn = tc.get("function") or {}
                    if not fn.get("name"):
                        continue
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    args_str = json.dumps(args, ensure_ascii=False)
                    if item_started:
                        yield _evt("response.output_text.done", {
                            "type": "response.output_text.done",
                            "item_id": item_id, "output_index": 0, "content_index": 0,
                            "text": "".join(text_buf),
                        })
                        yield _evt("response.content_part.done", {
                            "type": "response.content_part.done",
                            "item_id": item_id, "output_index": 0, "content_index": 0,
                            "part": {"type": "output_text", "text": "".join(text_buf), "annotations": []},
                        })
                        yield _evt("response.output_item.done", {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {"id": item_id, "type": "message", "role": "assistant",
                                     "status": "completed",
                                     "content": [{"type": "output_text", "text": "".join(text_buf), "annotations": []}]},
                        })
                        item_started = False
                    fc_id = tc.get("id") or f"fc_{uuid.uuid4().hex[:8]}"
                    yield _evt("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": 1,
                        "item": {"id": fc_id, "type": "function_call", "status": "in_progress",
                                 "call_id": fc_id, "name": fn.get("name", ""), "arguments": args_str},
                    })
                    yield _evt("response.output_item.done", {
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": {"id": fc_id, "type": "function_call", "status": "completed",
                                 "call_id": fc_id, "name": fn.get("name", ""), "arguments": args_str},
                    })
        # 收尾
        if item_started:
            full = "".join(text_buf)
            yield _evt("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": item_id, "output_index": 0, "content_index": 0, "text": full,
            })
            yield _evt("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": item_id, "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": full, "annotations": []},
            })
            yield _evt("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"id": item_id, "type": "message", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": full, "annotations": []}]},
            })
        elif not created:
            yield _evt("response.output_item.added", {
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": item_id, "type": "message", "role": "assistant",
                         "status": "completed",
                         "content": [{"type": "output_text", "text": "", "annotations": []}]},
            })
        # completed
        usage = {"input_tokens": 0, "output_tokens": len("".join(text_buf)) // 4, "total_tokens": 0}
        yield _evt("response.completed", {
            "type": "response.completed",
            "response": {
                "id": resp_id, "object": "response", "created_at": int(time.time()),
                "status": "completed", "model": model,
                "output": [{
                    "id": item_id, "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "".join(text_buf), "annotations": []}],
                }],
                "usage": usage,
            },
        })

    return _gen()
