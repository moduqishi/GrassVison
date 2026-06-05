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
from app.image_utils import extract_images_from_messages, remove_image_content
from app.providers import get_source_client
from app.schemas import ChatCompletionRequest, EnhancedModelConfig
from app.stats import get_stats
from app.vision import analyze_images, prepare_enhanced_messages


def _find_model(model_id: str) -> EnhancedModelConfig:
    """Find and validate an enhanced model by its public ID."""
    cfg = get_config()
    model = cfg.models.get(model_id)
    if not model or not model.enabled:
        raise ModelNotFoundError(model_id)
    return model


def _build_source_body(request: ChatCompletionRequest, model: EnhancedModelConfig) -> dict:
    body = {
        "model": model.source_model,
        "messages": [m.model_dump(exclude_none=True) for m in request.messages],
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
    """Try extracting token usage from a non-[DONE] SSE chunk."""
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
    stats = get_stats()

    images = extract_images_from_messages(
        [m.model_dump(exclude_none=True) for m in request.messages]
    )
    has_images = len(images) > 0 and model.vision_enabled
    image_count = len(images)

    vision_used = False
    vision_success = False
    vision_token_usage = None
    source_token_usage = None

    if not has_images:
        response, source_token_usage = await _forward_text(request, model)
    else:
        if len(images) > cfg.image.max_images:
            return _openai_error_response(400, f"Too many images: {len(images)} > {cfg.image.max_images}")

        image_urls = [img["url"] for img in images]
        vision_used = True

        try:
            vision_ctx = await analyze_images(
                messages=[m.model_dump(exclude_none=True) for m in request.messages],
                image_urls=image_urls,
                vision_provider_key=model.vision_provider,
                vision_model=model.vision_model,
                vision_prompt=model.vision_prompt,
                request_client=getattr(raw_request, "_httpx_client", None),
            )
            vision_success = True
            vision_token_usage = vision_ctx.get("token_usage") or {}
        except VisionAnalysisError as e:
            if model.vision_failure_mode == "skip":
                response, source_token_usage = await _forward_text(request, model)
                stats.record_call(
                    model=request.model, images=image_count,
                    stream=request.stream, elapsed=0,
                    vision_used=True, vision_success=False,
                )
                return response
            return _openai_error_response(502, f"Vision analysis failed: {e.message}")

        enhanced_messages = prepare_enhanced_messages(
            messages=[m.model_dump(exclude_none=True) for m in request.messages],
            vision_result=vision_ctx["result"],
        )

        body = _build_source_body(request, model)
        body["messages"] = enhanced_messages
        response, source_token_usage = await _forward_to_source(
            body=body,
            provider_key=model.source_provider,
            public_model_id=request.model if model.replace_response_model else model.source_model,
            stream=request.stream,
        )

    # Record usage statistics
    elapsed_total = time.time()  # approximate
    stats.record_call(
        model=request.model, images=image_count,
        stream=request.stream, elapsed=0,
        vision_used=vision_used, vision_success=vision_success,
        vision_tokens=vision_token_usage,
        source_tokens=source_token_usage,
    )

    return response


async def _forward_text(
    request: ChatCompletionRequest,
    model: EnhancedModelConfig,
) -> tuple[JSONResponse | StreamingResponse, dict | None]:
    body = _build_source_body(request, model)
    return await _forward_to_source(
        body=body,
        provider_key=model.source_provider,
        public_model_id=request.model if model.replace_response_model else model.source_model,
        stream=request.stream,
    )


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

    # Streaming: capture usage from the last chunk
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
                        # Track usage from last non-DONE chunk
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
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        ),
        usage_holder[0],  # may be None for streaming
    )
