"""Core proxy: chat completions routing, streaming, and vision enhancement."""
from __future__ import annotations

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

    if not all_images or not model.vision_enabled:
        # No images or vision disabled → forward directly
        body = _build_source_body(request, model, messages_raw)
        resp, source_usage = await _forward_to_source(
            body=body,
            provider_key=model.source_provider,
            public_model_id=request.model if model.replace_response_model else model.source_model,
            stream=request.stream,
        )
        stats_tracker.record_call(
            model=request.model, images=0,
            stream=request.stream, elapsed=0,
            vision_used=False, vision_success=False,
            source_tokens=source_usage,
        )
        return resp

    # ── 2. Determine which images can trigger NEW analysis ───────
    current_positions = extract_images_from_last_user_message(messages_raw)

    # Validate image count
    if len(all_images) > cfg.image.max_images:
        return _openai_error_response(400, f"Too many images: {len(all_images)} > {cfg.image.max_images}")

    # ── 3. Resolve all image descriptions (cache + vision calls) ─
    try:
        descriptions = await resolve_image_descriptions(
            images=all_images,
            model_config=model,
            allow_analysis_positions=current_positions,
            historical_cache_miss=cfg.image.historical_cache_miss,
            request_client=getattr(raw_request, "_httpx_client", None),
            user_question=extract_user_question(messages_raw),
        )
    except VisionAnalysisError as e:
        if model.vision_failure_mode == "skip":
            body = _build_source_body(request, model, messages_raw)
            resp, source_usage = await _forward_to_source(
                body=body,
                provider_key=model.source_provider,
                public_model_id=request.model if model.replace_response_model else model.source_model,
                stream=request.stream,
            )
            stats_tracker.record_call(
                model=request.model, images=len(all_images),
                stream=request.stream, elapsed=0,
                vision_used=True, vision_success=False,
                source_tokens=source_usage,
            )
            return resp
        return _openai_error_response(502, f"Vision analysis failed: {e.message}")

    # ── 4. Inject descriptions into messages ─────────────────────
    enhanced_messages = inject_image_descriptions(messages_raw, descriptions)

    # ── 5. Merge and inject vision context into last user msg ────
    merged = _merge_and_number_descriptions(descriptions)
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

    # ── 6. Assert no image_url blocks remain ────────────────────
    assert_no_image_url_blocks(enhanced_messages)

    # ── 7. Forward to source model ──────────────────────────────
    body = _build_source_body(request, model, enhanced_messages)
    resp, source_usage = await _forward_to_source(
        body=body,
        provider_key=model.source_provider,
        public_model_id=request.model if model.replace_response_model else model.source_model,
        stream=request.stream,
    )

    stats_tracker.record_call(
        model=request.model, images=len(all_images),
        stream=request.stream, elapsed=0,
        vision_used=True, vision_success=True,
        source_tokens=source_usage,
    )

    return resp


async def _forward_to_source(
    body: dict,
    provider_key: str,
    public_model_id: str,
    stream: bool = False,
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

    usage_holder: list[dict | None] = [None]

    async def _stream_with_tracking():
        first_chunk = True
        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_data = json.dumps({
                        "error": {
                            "message": error_body.decode()[:500],
                            "type": "upstream_error",
                            "code": resp.status_code,
                        }
                    })
                    yield f"data: {error_data}\n\n"
                    yield "data: [DONE]\n\n"
                    return
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
                        if u:
                            usage_holder[0] = u
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

    return (
        StreamingResponse(
            _stream_with_tracking(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        ),
        usage_holder[0],
    )
