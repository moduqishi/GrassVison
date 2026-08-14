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

    # 历史缓存复用需要全部图片位置（含历史轮次）
    all_images = extract_all_images_with_positions(messages_raw)

    # 预提示（可选）：默认关闭——首帧即视觉模型真实输出，思考链完全真实；
    # 开启则先推一条"正在处理"占位，消除下载/排队期的静默等待。
    if cfg.image.vision_stream_prelude:
        prelude, is_first["value"] = _build_vision_frame(
            "【正在处理用户发送的图片…】", public_model_id, stream_id, True
        )
        yield prelude

    # ── 阶段 1：流式视觉分析 ───────────────────────────────────
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
                stream_queue=queue,
                failure_mode=model.vision_failure_mode,
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
