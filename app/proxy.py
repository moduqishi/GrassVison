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
    extract_images_from_last_user_message,
    inject_image_descriptions,
    assert_no_image_url_blocks,
    extract_user_question,
)
from app.providers import get_source_client
from app.schemas import ChatCompletionRequest, EnhancedModelConfig
from app.stats import get_stats
from app.vision import resolve_image_descriptions, _merge_and_number_descriptions, _build_injection_text


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

    # ── 1.5 只处理当前（最后一条用户消息）里的图片 ──────────────
    # 历史消息里的图片一律剥离：不分析、不注入、不参与流式思考。
    # 解决：无关图片污染上下文、模型反复提及旧图、纯文字追问也被拖慢首 token。
    current_positions = extract_images_from_last_user_message(messages_raw)
    current_images = [img for img in all_images if img.position in current_positions]

    if not current_images or not model.vision_enabled:
        # 无当前图片或视觉关闭 → 剥离所有图片块后直接转发
        stripped = messages_raw
        if all_images:
            stripped = inject_image_descriptions(messages_raw, {})
        body = _build_source_body(request, model, stripped)
        if request.stream:
            def _on_noimg_usage(usage):
                stats_tracker.record_call(
                    model=request.model, images=len(all_images),
                    stream=True, elapsed=0,
                    vision_used=False, vision_success=False,
                    source_tokens=usage,
                )
            resp, _ = await _forward_to_source(
                body=body,
                provider_key=model.source_provider,
                public_model_id=request.model if model.replace_response_model else model.source_model,
                stream=True,
                on_usage=_on_noimg_usage,
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
                vision_used=False, vision_success=False,
                source_tokens=source_usage,
            )
        return resp

    # Validate image count（仅当前图片）
    if len(current_images) > cfg.image.max_images:
        return _openai_error_response(400, f"Too many images: {len(current_images)} > {cfg.image.max_images}")

    # ── 2.5 流式透传视觉思考：不阻塞等待分析，先流式显示视觉模型的
    #     思考/分析过程，再无缝衔接源模型（系统设置 image.stream_vision_thinking）─
    if request.stream and cfg.image.stream_vision_thinking:
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

    # ── 3. Resolve 当前图片的描述（缓存 + 视觉调用）────────────
    try:
        descriptions, vision_usage = await resolve_image_descriptions(
            images=current_images,
            model_config=model,
            allow_analysis_positions=current_positions,
            historical_cache_miss=cfg.image.historical_cache_miss,
            request_client=getattr(raw_request, "_httpx_client", None),
            user_question=extract_user_question(messages_raw),
        )
    except VisionAnalysisError as e:
        if model.vision_failure_mode == "skip":
            body = _build_source_body(request, model, messages_raw)
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
    # 只注入当前图片的描述（历史图片已剥离，不会进入当前问题）
    merged = _merge_and_number_descriptions(descriptions)
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
):
    """流式转发源模型：逐行 yield SSE 内容；收到 usage chunk 时回调 on_usage。"""
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
    finally:
        await client.aclose()


async def _combined_stream(
    request: ChatCompletionRequest,
    model: EnhancedModelConfig,
    messages_raw: list[dict],
    current_images: list,
    current_positions: set,
    raw_request: Request,
):
    """流式透传视觉模型的思考/分析过程，再无缝衔接源模型的思考与回答（单条 SSE 流）。

    阶段 1：视觉分析调用改为流式，增量以 reasoning_content 形式推给客户端，
            用户发完图立即能看到图像模型的思考链，不再静默等待。
    阶段 2：图片描述注入完成后，直接衔接源模型流式转发（思考链 + 回答）。
    """
    cfg = get_config()
    stats_tracker = get_stats()
    public_model_id = request.model if model.replace_response_model else model.source_model
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    queue: asyncio.Queue = asyncio.Queue()
    is_first = {"value": True}

    # 立即推一条预提示，消除图片下载/视觉模型首字前的静默等待
    prelude, is_first["value"] = _build_vision_frame(
        "【正在分析用户发送的图片…】", public_model_id, stream_id, True
    )
    yield prelude

    # ── 阶段 1：流式视觉分析 ───────────────────────────────────
    async def _resolve():
        try:
            return await resolve_image_descriptions(
                images=current_images,
                model_config=model,
                allow_analysis_positions=current_positions,
                historical_cache_miss=cfg.image.historical_cache_miss,
                request_client=getattr(raw_request, "_httpx_client", None),
                user_question=extract_user_question(messages_raw),
                stream_queue=queue,
            )
        finally:
            # 保证消费端一定能等到结束标记（即使内部抛异常也不会挂起）
            await queue.put(("done", "", ""))

    try:
        task = asyncio.create_task(_resolve())
        while True:
            item = await queue.get()
            if item[0] == "done":
                break
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
        # skip：直接用原始消息转发源模型
        body = _build_source_body(request, model, messages_raw)
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
    merged = _merge_and_number_descriptions(descriptions)
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

    async for line in _iter_source_stream(body, model.source_provider, public_model_id, on_usage=_on_source_usage):
        yield line


async def _forward_to_source(
    body: dict,
    provider_key: str,
    public_model_id: str,
    stream: bool = False,
    on_usage: callable | None = None,
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
            if "model" in data:
                data["model"] = public_model_id
            return JSONResponse(content=data), usage
        except httpx.TimeoutException:
            return _openai_error_response(504, "Source model timeout"), None
        finally:
            await client.aclose()

    async def _stream_with_tracking():
        async for line in _iter_source_stream(body, provider_key, public_model_id, on_usage=on_usage):
            yield line

    return (
        StreamingResponse(
            _stream_with_tracking(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        ),
        None,
    )
