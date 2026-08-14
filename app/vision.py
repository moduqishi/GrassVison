"""Vision analysis: prompt loading, vision model calling, caching, and result injection."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import time
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from app.config import get_config, PROMPTS_DIR
from app.errors import VisionAnalysisError
from app.image_cache import CacheEntry, get_image_cache
from app.image_utils import (
    ExtractedImage, ImagePosition,
    compute_content_hash, preprocess_image,
    extract_user_question, resolve_image_to_base64,
    DATA_URL_RE, PREPROCESS_VERSION,
)
from app.providers import get_vision_client

logger = logging.getLogger("grassvision")

# 视觉分析失败结果的缓存时长（秒）：短 TTL 防止故障期间下游请求对渠道雪崩重试
FAILURE_CACHE_TTL_SECONDS = 60

# 多图联合分析的对比意图关键词（multi_image_mode=auto 时命中则合并为一次调用）
_COMPARISON_KEYWORDS = (
    "对比", "比较", "区别", "差异", "不同", "相同", "哪个更", "哪个好", "哪张",
    "哪一个", "谁更", "找不同", "diff", "compare", "similar", "which one",
    "which is", "better", "prefer", "vs", "versus",
)


def _detect_comparison_intent(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in _COMPARISON_KEYWORDS)


async def _call_vision_chain(
    provider_ids: list[str],
    explicit_model: str,
    system_prompt: str,
    user_question: str,
    image_urls: list[str],
    request_client: httpx.AsyncClient | None = None,
    stream_queue: asyncio.Queue | None = None,
) -> dict:
    """按序尝试 provider_ids（主渠道 + 故障转移链），返回第一个成功的结果。

    全部失败时抛出最后一个 VisionAnalysisError（错误信息含已尝试渠道）。
    """
    last_error: VisionAnalysisError | None = None
    for pid in provider_ids:
        try:
            if stream_queue is not None:
                async def _emit(kind: str, text: str) -> None:
                    await stream_queue.put(("token", kind, text))
                return await _call_vision_model_stream(
                    provider_id=pid,
                    model_id=explicit_model,
                    system_prompt=system_prompt,
                    user_question=user_question,
                    image_urls=image_urls,
                    request_client=request_client,
                    emit=_emit,
                )
            return await _call_vision_model(
                provider_id=pid,
                model_id=explicit_model,
                system_prompt=system_prompt,
                user_question=user_question,
                image_urls=image_urls,
                request_client=request_client,
            )
        except VisionAnalysisError as e:
            last_error = e
    raise last_error or VisionAnalysisError("No vision provider available")


# ── 定位-放大-再读（grounding）与长截图切片 ─────────────────────

_GROUNDING_KEYWORDS = (
    "按钮", "输入框", "图标", "链接", "菜单", "标签", "图片", "标题", "元素",
    "坐标", "位置", "点击", "哪里", "哪个", "第几", "区域", "框", "控件",
    "button", "click", "where", "which", "element", "icon", "input", "box",
)

_GROUNDING_BOX_RE = re.compile(
    r"x1:\s*(-?\d+)[^\d-]*y1:\s*(-?\d+)[^\d-]*x2:\s*(-?\d+)[^\d-]*y2:\s*(-?\d+)",
    re.IGNORECASE,
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """剥离视觉模型内容中的 <think>…</think> 思考块（如 MiniMax 非流式返回）。

    思考过程不应作为"图片描述"注入给源模型，且其中的近似坐标可能干扰 grounding 解析。
    """
    if not text:
        return ""
    return _THINK_BLOCK_RE.sub("", text).strip()

LONG_SCREENSHOT_MIN_RATIO = 3.0   # 高/宽 达到该比例视为长截图
LONG_SCREENSHOT_MIN_HEIGHT = 1800  # 且高度至少这么多像素
SLICE_BAND_HEIGHT = 1000           # 每段高度
SLICE_OVERLAP = 120                # 相邻段重叠像素


def _has_grounding_intent(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in _GROUNDING_KEYWORDS)


def _parse_grounding_box(text: str) -> tuple[int, int, int, int] | None:
    """从定位结果解析 0-1000 归一化框；无有效框返回 None。"""
    m = _GROUNDING_BOX_RE.search(text or "")
    if not m:
        return None
    x1, y1, x2, y2 = (int(g) for g in m.groups())
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > 1000 or y2 > 1000:
        return None
    return (x1, y1, x2, y2)


def _image_dimensions(raw_bytes: bytes) -> tuple[int, int] | None:
    try:
        return Image.open(io.BytesIO(raw_bytes)).size
    except Exception:
        return None


def _crop_zoom_data_url(raw_bytes: bytes, box: tuple[int, int, int, int], scale: int = 2) -> str | None:
    """按 0-1000 归一化框裁剪原图并放大，返回 JPEG data URL；失败返回 None。"""
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        x1 = max(0, int(box[0] * w / 1000))
        y1 = max(0, int(box[1] * h / 1000))
        x2 = min(w, int(box[2] * w / 1000))
        y2 = min(h, int(box[3] * h / 1000))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _slice_data_url(raw_bytes: bytes, y1: int, y2: int) -> str | None:
    """从原图裁剪一段水平带，返回 JPEG data URL；失败返回 None。"""
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        y2 = min(y2, h)
        crop = img.crop((0, y1, w, y2))
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _structure_evidence(text: str) -> str | None:
    """把视觉模型返回的结构化证据 JSON 转成易引用的注入文本。

    - 容忍 markdown 代码围栏与前后杂文，只提取 JSON 对象。
    - 字段缺失/类型不对时尽量降级（保留能用的部分）。
    - 解析失败返回 None（上层保留原始文本）。
    """
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return None

    summary = str(data.get("summary", "") or "").strip()
    ocr = data.get("ocr") if isinstance(data.get("ocr"), dict) else {}
    full_text = str(ocr.get("full_text", "") or "").strip()
    layout = data.get("layout") if isinstance(data.get("layout"), dict) else {}
    regions = layout.get("regions") if isinstance(layout.get("regions"), list) else []
    semantics = data.get("semantics") if isinstance(data.get("semantics"), dict) else {}
    entities = semantics.get("entities") if isinstance(semantics.get("entities"), list) else []
    visual = data.get("visual") if isinstance(data.get("visual"), dict) else {}
    colors = visual.get("dominant_colors") if isinstance(visual.get("dominant_colors"), list) else []
    style = str(visual.get("style", "") or "").strip()
    uncertainty = data.get("uncertainty") if isinstance(data.get("uncertainty"), list) else []

    parts: list[str] = []
    if summary:
        parts.append(f"【摘要】\n{summary}")
    if full_text:
        parts.append(f"【全文文字】\n{full_text}")
    region_lines = []
    for r in regions:
        if not isinstance(r, dict):
            continue
        rtype = str(r.get("type", "") or "").strip()
        rtext = str(r.get("text", "") or "").strip()
        if rtype or rtext:
            region_lines.append(f"- [{rtype or '区块'} {r.get('reading_order', '')}] {rtext}".rstrip())
    if region_lines:
        parts.append("【版面结构】\n" + "\n".join(region_lines))
    entity_lines = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "") or "").strip()
        etype = str(e.get("type", "") or "").strip()
        ev = str(e.get("evidence", "") or "").strip()
        if name:
            entity_lines.append(f"- {name}（{etype or '未知类型'}）{'：' + ev if ev else ''}")
    if entity_lines:
        parts.append("【关键实体】\n" + "\n".join(entity_lines))
    visual_lines = []
    if colors:
        visual_lines.append("主色: " + ", ".join(str(c) for c in colors if isinstance(c, str)))
    if style:
        visual_lines.append(f"风格: {style}")
    if visual_lines:
        parts.append("【视觉特征】\n" + "\n".join(visual_lines))
    if uncertainty:
        parts.append("【不确定项 ⚠️】\n" + "\n".join(f"- {u}" for u in uncertainty if isinstance(u, str)))
    if not parts:
        return None
    return "\n\n".join(parts)


async def _resolve_long_screenshot(
    raw_bytes: bytes,
    provider_ids: list[str],
    explicit_model: str,
    system_prompt: str,
    request_client: httpx.AsyncClient | None,
) -> str | None:
    """长截图分带 OCR：检测高宽比 → 逐段分析 → 合并。失败返回 None（保留原结果）。"""
    dims = _image_dimensions(raw_bytes)
    if not dims:
        return None
    w, h = dims
    if h < LONG_SCREENSHOT_MIN_HEIGHT or h / max(w, 1) < LONG_SCREENSHOT_MIN_RATIO:
        return None
    stride = SLICE_BAND_HEIGHT - SLICE_OVERLAP
    bands: list[tuple[int, int]] = []
    y = 0
    while y < h:
        y2 = min(h, y + SLICE_BAND_HEIGHT)
        bands.append((y, y2))
        y += stride
    if len(bands) < 2:
        return None
    parts: list[str] = []
    for i, (y1b, y2b) in enumerate(bands, 1):
        data_url = _slice_data_url(raw_bytes, y1b, y2b)
        if data_url is None:
            return None
        result = await _call_vision_chain(
            provider_ids=provider_ids,
            explicit_model=explicit_model,
            system_prompt=system_prompt,
            user_question="请完整提取这一部分截图中的文字、代码、界面元素与布局结构，保持顺序与格式，不要遗漏。",
            image_urls=[data_url],
            request_client=request_client,
        )
        parts.append(f"【第 {i} 段（原图 y={y1b}-{y2b}px）】\n{result['result']}")
    return (
        f"（该截图较长，已按 {len(bands)} 段分段分析，相邻段有 {SLICE_OVERLAP}px 重叠，"
        f"重复内容属正常）\n\n" + "\n\n".join(parts)
    )


async def _resolve_grounding_zoom(
    raw_bytes: bytes,
    data_url: str,
    provider_ids: list[str],
    explicit_model: str,
    grounding_prompt: str,
    system_prompt: str,
    user_question: str,
    request_client: httpx.AsyncClient | None,
) -> str | None:
    """定位-放大-再读：先定位目标元素坐标框，再裁剪放大做二次精读。失败返回 None。"""
    if not _has_grounding_intent(user_question):
        return None
    first = await _call_vision_chain(
        provider_ids=provider_ids,
        explicit_model=explicit_model,
        system_prompt=grounding_prompt,
        user_question=user_question,
        image_urls=[data_url],
        request_client=request_client,
    )
    box = _parse_grounding_box(first["result"])
    if box is None:
        return None
    zoom_url = _crop_zoom_data_url(raw_bytes, box)
    if zoom_url is None:
        return None
    second = await _call_vision_chain(
        provider_ids=provider_ids,
        explicit_model=explicit_model,
        system_prompt=system_prompt,
        user_question=user_question,
        image_urls=[zoom_url],
        request_client=request_client,
    )
    x1, y1, x2, y2 = box
    return (
        f"[已定位目标元素，0-1000 归一化坐标框 x1={x1}, y1={y1}, x2={x2}, y2={y2}]\n"
        f"{first['result']}\n\n"
        f"【目标元素放大分析】\n{second['result']}"
    )


def _parse_region(region: str) -> tuple[int, int, int, int] | None:
    """解析工具参数 region "x1,y1,x2,y2"（0-1000 归一化）。"""
    try:
        parts = [int(p.strip()) for p in region.split(",")]
        if len(parts) != 4:
            return None
        x1, y1, x2, y2 = parts
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > 1000 or y2 > 1000:
            return None
        return (x1, y1, x2, y2)
    except (ValueError, AttributeError):
        return None


async def reexamine_image(
    url: str,
    question: str,
    region: str | None,
    model_config,
    request_client: httpx.AsyncClient | None = None,
    emit: callable | None = None,
    usage_accumulator: dict | None = None,
) -> str:
    """协议化服务端重看（grassvision_view_image 工具执行体）。

    用请求内的图片（data URL 直接解码 / HTTP URL 重新下载）+ 新意图重新分析；
    region 提供 0-1000 归一化区域时，先本地裁剪放大再精读（工具化 grounding）。
    emit 提供时改为流式视觉调用（增量经 emit(kind, text) 回调，供流式融合版推送思考链）。
    usage_accumulator 提供时累计本次重看的视觉 token 用量（供用量统计）。
    """
    if not (question or "").strip():
        raise VisionAnalysisError("grassvision_view_image 缺少 question 参数")
    cfg = get_config()
    data_url = await resolve_image_to_base64(url, request_client)

    provider_ids = [model_config.vision_provider]
    provider_ids += [p for p in (getattr(model_config, "vision_provider_failover", None) or []) if p]
    provider_ids = list(dict.fromkeys(provider_ids))
    explicit_model = model_config.vision_model or ""
    system_prompt = _load_prompt_content(model_config.vision_prompt, question)

    target = data_url
    box = _parse_region(region) if region else None
    if box is not None:
        import base64 as _b64
        m = DATA_URL_RE.match(data_url)
        if m:
            raw = _b64.b64decode(m.group(2))
            zoom = _crop_zoom_data_url(raw, box)
            if zoom:
                target = zoom
                x1, y1, x2, y2 = box
                system_prompt += (
                    f"\n（已按请求裁剪放大 0-1000 区域 x1={x1},y1={y1},x2={x2},y2={y2}，"
                    f"只分析该区域）"
                )

    if emit is not None:
        result = await _call_vision_model_stream(
            provider_id=provider_ids[0],
            model_id=explicit_model,
            system_prompt=system_prompt,
            user_question=question,
            image_urls=[target],
            request_client=request_client,
            emit=emit,
        )
    else:
        result = await _call_vision_chain(
            provider_ids=provider_ids,
            explicit_model=explicit_model,
            system_prompt=system_prompt,
            user_question=question,
            image_urls=[target],
            request_client=request_client,
        )
    # 重看也是一次实际视觉调用（计入缓存统计的 vision_calls）
    try:
        get_image_cache().record_vision_call()
    except Exception:
        pass
    # 累计重看的视觉 token 用量（供用量统计）
    if usage_accumulator is not None and result.get("token_usage"):
        for k, v in result["token_usage"].items():
            if isinstance(v, (int, float)):
                usage_accumulator[k] = usage_accumulator.get(k, 0) + v
    return result["result"]


def build_cache_key(
    content_hash: str,
    provider_id: str,
    model_id: str,
    prompt_hash: str,
    analysis_mode: str = "independent",
    prep_version: str = PREPROCESS_VERSION,
) -> str:
    raw = "|".join([content_hash, provider_id, model_id, prompt_hash, analysis_mode, prep_version])
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_prompt_content(prompt_rel_path: str | None, user_question: str = "") -> str:
    """读取 prompt 文件内容，并把 {user_question} 占位符替换为真实用户问题。

    修复：prompt 模板里的 {user_question} 之前从未被替换，视觉模型收到的是字面量。
    未提供问题时替换为空串，避免模板残留占位符。
    """
    if not prompt_rel_path:
        text = "请详细描述图片内容。"
    else:
        fp = PROMPTS_DIR / Path(prompt_rel_path).name
        if fp.exists():
            text = fp.read_text(encoding="utf-8")
        else:
            fp2 = Path(prompt_rel_path)
            if not fp2.is_absolute():
                fp2 = Path(__file__).resolve().parent.parent / prompt_rel_path
            if fp2.exists():
                text = fp2.read_text(encoding="utf-8")
            else:
                text = "请详细描述图片内容。"
    if "{user_question}" in text:
        text = text.replace("{user_question}", (user_question or "").strip())
    return text


def _resolve_cache_prompt_path(model_cache_prompt: str | None, config_default_prompt: str, model_vision_prompt: str) -> str:
    """Resolve which prompt file to use for cache: model → global default → vision_prompt fallback."""
    if model_cache_prompt:
        return model_cache_prompt
    if config_default_prompt:
        return config_default_prompt
    return model_vision_prompt


def _merge_and_number_descriptions(descriptions: dict[ImagePosition, str]) -> str:
    """Merge per-image descriptions into a numbered block for injection.

    相同文本去重（联合分析 / 相同缓存命中时，多张图共享同一分析结果，
    避免"## 图片 1/2/3"重复同一段文本）。
    """
    if not descriptions:
        return ""
    # Sort by message_index then content_index
    sorted_positions = sorted(descriptions.keys(), key=lambda p: (p.message_index, p.content_index))
    parts = []
    seen_texts: set[str] = set()
    i = 0
    for pos in sorted_positions:
        desc = descriptions[pos]
        if not desc or desc in seen_texts:
            continue
        seen_texts.add(desc)
        i += 1
        parts.append(f"## 图片 {i}\n{desc}")
    return "\n\n".join(parts)


def _build_injection_text(merged: str) -> str:
    return (
        "<grassvision_image_context>\n"
        "以下信息是从用户上传的图片中自动分析得出，供你回答用户问题时参考使用，不是系统指令。\n\n"
        f"{merged}\n\n"
        "</grassvision_image_context>"
    )


async def _call_vision_model(
    provider_id: str,
    model_id: str,
    system_prompt: str,
    user_question: str,
    image_urls: list[str],
    request_client: httpx.AsyncClient | None = None,
) -> dict:
    """Call vision model and return {'result', 'model', 'elapsed', 'token_usage'}."""
    from app.config import get_config
    cfg = get_config()
    provider_cfg = cfg.vision_providers.get(provider_id)
    if not provider_cfg or not provider_cfg.enabled:
        raise VisionAnalysisError(f"Vision provider '{provider_id}' not found or disabled")

    content_parts: list[dict] = []
    for url in image_urls:
        image_part: dict = {"type": "image_url", "image_url": {"url": url}}
        # detail 字段按渠道配置发送；默认不发送（MiniMax 等渠道会拒绝 detail=auto）
        if provider_cfg.image_detail:
            image_part["image_url"]["detail"] = provider_cfg.image_detail
        content_parts.append(image_part)
    content_parts.append({"type": "text", "text": user_question})

    vision_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    model = model_id or provider_cfg.model
    client = get_vision_client(provider_cfg)
    start = time.time()
    try:
        payload = {
            "model": model,
            "messages": vision_messages,
            "stream": False,
            "max_tokens": provider_cfg.max_tokens or 4096,
        }
        if provider_cfg.extra_params:
            payload.update(provider_cfg.extra_params)
        resp = await client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            raise VisionAnalysisError(f"Vision model returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        result_text = _strip_think_blocks(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        elapsed = time.time() - start
        return {
            "result": result_text,
            "model": model,
            "elapsed": elapsed,
            "token_usage": data.get("usage"),
        }
    except httpx.TimeoutException:
        raise VisionAnalysisError(f"Vision model request timed out after {provider_cfg.timeout}s")
    # 连接池化：client 由 providers 池统一管理，不在调用点关闭


async def _call_vision_model_stream(
    provider_id: str,
    model_id: str,
    system_prompt: str,
    user_question: str,
    image_urls: list[str],
    request_client: httpx.AsyncClient | None = None,
    emit: callable | None = None,
) -> dict:
    """流式调用视觉模型：边接收边通过 emit(kind, text) 透传增量（kind ∈ {"reasoning", "content"}）。
    返回 {'result', 'model', 'elapsed', 'token_usage'}，与 _call_vision_model 一致。"""
    from app.config import get_config
    cfg = get_config()
    provider_cfg = cfg.vision_providers.get(provider_id)
    if not provider_cfg or not provider_cfg.enabled:
        raise VisionAnalysisError(f"Vision provider '{provider_id}' not found or disabled")

    content_parts: list[dict] = []
    for url in image_urls:
        image_part: dict = {"type": "image_url", "image_url": {"url": url}}
        if provider_cfg.image_detail:
            image_part["image_url"]["detail"] = provider_cfg.image_detail
        content_parts.append(image_part)
    content_parts.append({"type": "text", "text": user_question})

    vision_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    model = model_id or provider_cfg.model
    client = get_vision_client(provider_cfg)
    start = time.time()

    async def _stream_fallback():
        """流式不可用（渠道不支持 stream / 响应无增量）时回退非流式调用，
        并把完整分析结果一次性放进思考链，保证视觉阶段始终可见。"""
        result = await _call_vision_model(
            provider_id=provider_id,
            model_id=model_id,
            system_prompt=system_prompt,
            user_question=user_question,
            image_urls=image_urls,
            request_client=request_client,
        )
        if emit and result.get("result"):
            await emit("content", result["result"])
        return result

    try:
        payload = {
            "model": model,
            "messages": vision_messages,
            "stream": True,
            "max_tokens": provider_cfg.max_tokens or 4096,
        }
        if provider_cfg.extra_params:
            payload.update(provider_cfg.extra_params)
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                # 该渠道可能不支持 stream=true：回退为非流式调用
                return await _stream_fallback()
            result_parts: list[str] = []
            token_usage: dict | None = None
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = ((data.get("choices") or [{}])[0].get("delta")) or {}
                # 兼容不同渠道的思考字段：reasoning_content / reasoning / thinking
                reasoning = (delta.get("reasoning_content")
                             or delta.get("reasoning")
                             or delta.get("thinking"))
                if reasoning:
                    if emit:
                        await emit("reasoning", reasoning)
                content = delta.get("content")
                if content:
                    result_parts.append(content)
                    if emit:
                        await emit("content", content)
                if data.get("usage"):
                    token_usage = data["usage"]
            if not result_parts:
                # 流式响应里没有产出任何内容增量：回退非流式，保证分析结果仍出现
                return await _stream_fallback()
            elapsed = time.time() - start
            return {
                "result": _strip_think_blocks("".join(result_parts)),
                "model": model,
                "elapsed": elapsed,
                "token_usage": token_usage,
            }
    except httpx.TimeoutException:
        raise VisionAnalysisError(f"Vision model request timed out after {provider_cfg.timeout}s")
    # 连接池化：client 由 providers 池统一管理，不在调用点关闭


async def resolve_image_descriptions(
    images: list[ExtractedImage],
    model_config,
    allow_analysis_positions: set[ImagePosition],
    historical_cache_miss: str = "analyze",
    request_client: httpx.AsyncClient | None = None,
    user_question: str = "",
    stream_queue: asyncio.Queue | None = None,
    failure_mode: str = "error",
) -> tuple[dict[ImagePosition, str], dict]:
    """
    Main entry point for resolving all image descriptions.
    Returns (descriptions, vision_usage) where:
      - descriptions: {ImagePosition: description_text} for every image position
      - vision_usage: accumulated token usage from all vision model calls

    - Only positions in allow_analysis_positions trigger NEW vision calls.
    - Historical images with cache misses are handled per historical_cache_miss.
    - Same cache_key is deduplicated within a single request.
    - 传入 stream_queue 时，新发起的视觉调用改为流式：每个增量以 ("token", kind, text)
      放入队列（kind ∈ {"reasoning", "content"}），全部处理结束后放入 ("done", "", "")。
    - 多图并发处理（信号量限制并发度）；问题感知模式（question_aware_cache）下
      {user_question} 被替换为真实问题且缓存键随问题变化。
    - failure_mode：当前图片的视觉调用失败时，error=抛出 VisionAnalysisError（上层 502），
      skip=注入失败说明后继续（fail-open）。
    """
    import base64

    cfg = get_config()
    cache = get_image_cache()
    descriptions: dict[ImagePosition, str] = {}
    vision_usage: dict = {}

    if not images:
        return descriptions, vision_usage

    question_aware = bool(getattr(cfg.image, "question_aware_cache", False))
    is_cache_enabled = cfg.image.vision_cache.enabled
    # 视觉 prompt 只有在「问题感知模式」或「缓存关闭」时才会真正发给视觉模型，
    # 这两种情况都值得注入真实用户问题（修复 {user_question} 占位符从不替换的 bug）。
    use_question = question_aware or not is_cache_enabled

    # Resolve prompts
    vision_prompt_path = model_config.vision_prompt
    cache_prompt_path = _resolve_cache_prompt_path(
        model_config.cache_prompt,
        cfg.image.vision_cache.default_prompt,
        vision_prompt_path,
    )
    vision_prompt_text = _load_prompt_content(vision_prompt_path, user_question if use_question else "")
    cache_prompt_text = _load_prompt_content(cache_prompt_path) if is_cache_enabled else vision_prompt_text

    # 实际发送给视觉模型的 system prompt 与用户问题：
    if is_cache_enabled and not question_aware:
        # 传统模式：固定缓存 prompt + 固定问题，缓存跨问题复用
        effective_prompt = cache_prompt_text
        vision_user_question = "请根据 system prompt 中的要求分析这张图片。"
    else:
        # 问题感知模式（或缓存关闭）：使用带真实问题的视觉 prompt
        effective_prompt = vision_prompt_text
        vision_user_question = user_question or "请分析这张图片的内容。"

    prompt_hash = hashlib.sha256(effective_prompt.encode()).hexdigest()

    # Group images by their raw bytes → cache_key (dedup)
    # image_key: (url hash) → list of ExtractedImage  (for dedup within request)
    url_results: dict[str, str] = {}  # url → cached or new result text
    url_to_cache_key: dict[str, str] = {}
    url_to_status: dict[str, str] = {}  # "cached" | "owner" | "waiter" | "error" | "dropped"
    url_to_images: dict[str, list[ExtractedImage]] = {}
    url_to_raw_bytes: dict[str, bytes] = {}  # url → 原始字节（供裁剪/缩放/切片）
    url_to_data_url: dict[str, str] = {}     # url → 已解析的 data URL（供二次视觉调用）

    for img in images:
        url_to_images.setdefault(img.url, []).append(img)

    total_bytes = 0
    sem = asyncio.Semaphore(max(1, int(getattr(cfg.image, "vision_concurrency", 4))))

    # 视觉渠道链：主渠道 + 故障转移渠道（去重、去空）
    primary_provider = model_config.vision_provider
    failover_providers = [p for p in (getattr(model_config, "vision_provider_failover", None) or []) if p]
    provider_chain = list(dict.fromkeys([primary_provider] + failover_providers))
    explicit_model = model_config.vision_model or ""

    # ── 联合分析（多图一次调用）：multi_image_mode=combined 或
    #    auto+对比意图 时，当前多张图片合并为一次视觉调用（可对比、省调用）。
    combined_urls: set[str] = set()
    combined_key: str = ""
    current_urls = list(dict.fromkeys(
        img.url for img in images if img.position in allow_analysis_positions
    ))
    multi_mode = getattr(cfg.image, "multi_image_mode", "independent") or "independent"
    use_combined = (
        len(current_urls) >= 2
        and (multi_mode == "combined" or (multi_mode == "auto" and _detect_comparison_intent(user_question)))
    )

    if use_combined:
        status = ""
        try:
            # 1) 并发下载全部当前图片，计算哈希与总大小
            async def _dl(url: str) -> tuple[str, str, int]:
                async with sem:
                    r = await resolve_image_to_base64(url, request_client)
                    m = DATA_URL_RE.match(r)
                    if not m:
                        raise VisionAnalysisError(f"图片解析失败: {url}")
                    return url, r, len(base64.b64decode(m.group(2)))

            downloaded = await asyncio.gather(*(_dl(u) for u in current_urls))
            resolved_map: dict[str, str] = {}
            content_hashes: list[str] = []
            combined_bytes = 0
            max_total_mb = getattr(cfg.image, "max_total_size_mb", 0) or 0
            for url, r, nbytes in downloaded:
                resolved_map[url] = r
                combined_bytes += nbytes
                if max_total_mb > 0 and combined_bytes > max_total_mb * 1024 * 1024:
                    raise VisionAnalysisError(f"图片总大小超限: 超过 {max_total_mb}MB")
                m = DATA_URL_RE.match(r)
                content_hashes.append(compute_content_hash(base64.b64decode(m.group(2))))

            # 2) 联合缓存键：图片哈希集合 + 渠道 + prompt（问题感知时含问题）
            combined_key = hashlib.sha256(
                ("|".join(sorted(content_hashes)) + "|" + primary_provider
                 + "|" + str(model_config.vision_model or "") + "|" + prompt_hash + "|combined").encode()
            ).hexdigest()

            # 3) 缓存查询 / 等待 / 发起一次调用
            entry, status = await cache.get_or_reserve(combined_key)
            if status == "cached" and entry:
                combined_text = entry.result
                # 缓存命中：流式模式下把缓存结果放入思考链（新生成时已通过 emit 流式推送，不重复）
                if stream_queue is not None and combined_text:
                    await stream_queue.put(("token", "content", combined_text))
            elif status == "waiter":
                combined_text = (await cache.wait_inflight(combined_key)).result
                if stream_queue is not None and combined_text:
                    await stream_queue.put(("token", "content", combined_text))
            else:
                n = len(current_urls)
                combined_question = (
                    f"请同时分析这 {n} 张图片并回答用户的问题（如需对比请逐项对比）：{user_question}"
                    if user_question
                    else f"请同时分析这 {n} 张图片，说明每张的内容以及它们之间的异同与关系。"
                )
                cache.record_vision_call()  # 联合分析：一次实际视觉调用
                result = await _call_vision_chain(
                    provider_ids=provider_chain,
                    explicit_model=explicit_model,
                    system_prompt=effective_prompt,
                    user_question=combined_question,
                    image_urls=[resolved_map[u] for u in current_urls],
                    request_client=request_client,
                    stream_queue=stream_queue,
                )
                combined_text = result["result"]
                if result.get("token_usage"):
                    for k, v in result["token_usage"].items():
                        if isinstance(v, (int, float)):
                            vision_usage[k] = vision_usage.get(k, 0) + v
                if is_cache_enabled:
                    now = time.monotonic()
                    await cache.set(combined_key, CacheEntry(
                        result=combined_text,
                        content_hash="|".join(sorted(content_hashes)),
                        provider_id=primary_provider,
                        model_id=str(model_config.vision_model or ""),
                        prompt_hash=prompt_hash,
                        analysis_mode="combined",
                        created_at=now,
                        expires_at=now + cfg.image.vision_cache.ttl_seconds,
                    ))

            # 4) 只把完整联合结果放在第一个当前位置，其余位置置空（注入时移除）
            combined_urls = set(current_urls)
            first = True
            for img in images:
                if img.position in allow_analysis_positions and img.url in combined_urls:
                    descriptions[img.position] = combined_text if first else ""
                    first = False
        except (VisionAnalysisError, Exception):
            # 联合分析失败（渠道不支持多图 / 下载失败等）→ 回退逐图分析
            combined_urls = set()
            # 若拿到了 owner 且调用失败，需要 resolve inflight（等待者不挂起）
            try:
                if status == "owner" and is_cache_enabled and combined_key:
                    await cache.set(combined_key, CacheEntry(
                        result="[分析失败]", content_hash="",
                        provider_id=primary_provider, model_id="",
                        prompt_hash=prompt_hash, analysis_mode="combined",
                        created_at=time.monotonic(), expires_at=time.monotonic(),
                    ))
            except Exception:
                pass

    async def _process_url(url: str, img_list: list[ExtractedImage]) -> None:
        """处理单个 URL：下载→哈希→缓存查询→(需要时)视觉调用。多图并发执行。"""
        nonlocal total_bytes
        async with sem:
            if url in combined_urls:
                # 已由联合分析处理（当前位置描述已就绪）
                return
            # Download + preprocess
            try:
                resolved_url = await resolve_image_to_base64(url, request_client)
            except Exception as e:
                # Download/parse failure → mark all positions as error
                url_results[url] = f"[图片加载失败: {e}]"
                url_to_status[url] = "error"
                return

            # Get raw bytes for hashing
            m = DATA_URL_RE.match(resolved_url)
            if not m:
                url_results[url] = "[图片解析失败]"
                url_to_status[url] = "error"
                return
            raw_bytes = base64.b64decode(m.group(2))
            url_to_raw_bytes[url] = raw_bytes
            url_to_data_url[url] = resolved_url

            # 请求内总大小限制（max_total_size_mb）
            max_total_mb = getattr(cfg.image, "max_total_size_mb", 0) or 0
            if max_total_mb > 0:
                total_bytes += len(raw_bytes)
                if total_bytes > max_total_mb * 1024 * 1024:
                    url_results[url] = f"[图片总大小超限: 超过 {max_total_mb}MB]"
                    url_to_status[url] = "error"
                    return

            # Compute content hash
            content_hash = compute_content_hash(raw_bytes)

            provider_id = model_config.vision_provider
            model_id = model_config.vision_model or cfg.vision_providers.get(provider_id, None)
            if hasattr(model_id, 'model'):
                model_id = model_id.model
            if not model_id:
                model_id = ""

            cache_key = build_cache_key(content_hash, provider_id, str(model_id), prompt_hash)
            url_to_cache_key[url] = cache_key

            # Check if any of these images can trigger new analysis
            can_analyze = any(img.position in allow_analysis_positions for img in img_list)
            is_historical = not can_analyze

            # Query cache — returns (entry_or_none, status_str)
            cached_entry, status_str = await cache.get_or_reserve(cache_key)

            if status_str == "cached" and cached_entry:
                url_results[url] = cached_entry.result
                url_to_status[url] = "cached"
                # 缓存命中：若处于流式透传模式，把缓存的分析结果也放进思考链，
                # 保证视觉阶段始终有可见内容（而不是只有一条预提示）。
                if stream_queue is not None and cached_entry.result:
                    await stream_queue.put(("token", "content", "（该图片此前已分析过，直接复用分析结果）\n" + cached_entry.result))
            elif status_str == "waiter":
                # Wait for inflight
                try:
                    entry = await cache.wait_inflight(cache_key)
                    url_results[url] = entry.result
                    url_to_status[url] = "cached"
                    if stream_queue is not None and entry.result:
                        await stream_queue.put(("token", "content", "（该图片此前已分析过，直接复用分析结果）\n" + entry.result))
                except Exception:
                    url_results[url] = "[图片分析超时]"
                    url_to_status[url] = "error"
            elif status_str == "owner":
                # We need to call vision model — but only if allowed
                if can_analyze or (is_historical and historical_cache_miss == "analyze"):
                    # 视觉调用（主渠道 + 故障转移链）——上报实际视觉调用次数
                    cache.record_vision_call()
                    try:
                        result = await _call_vision_chain(
                            provider_ids=provider_chain,
                            explicit_model=explicit_model,
                            system_prompt=effective_prompt,
                            user_question=vision_user_question,
                            image_urls=[resolved_url],
                            request_client=request_client,
                            stream_queue=stream_queue,
                        )
                        url_results[url] = result["result"]
                        url_to_status[url] = "new"

                        # Accumulate vision token usage
                        if result.get("token_usage"):
                            for k, v in result["token_usage"].items():
                                if isinstance(v, (int, float)):
                                    vision_usage[k] = vision_usage.get(k, 0) + v

                        # Write to cache if enabled
                        if is_cache_enabled:
                            now = time.monotonic()
                            entry = CacheEntry(
                                result=result["result"],
                                content_hash=content_hash,
                                provider_id=provider_id,
                                model_id=str(model_id),
                                prompt_hash=prompt_hash,
                                analysis_mode="independent",
                                created_at=now,
                                expires_at=now + cfg.image.vision_cache.ttl_seconds,
                            )
                            await cache.set(cache_key, entry)
                    except VisionAnalysisError as e:
                        # 写失败缓存并 resolve inflight（等待者拿到失败标记而非永久挂起）。
                        # error 模式：立即过期（下次请求重试，保持"大声失败"语义）；
                        # skip 模式：短 TTL 抑制故障期间对渠道的雪崩重试。
                        now = time.monotonic()
                        failure_ttl = FAILURE_CACHE_TTL_SECONDS if failure_mode == "skip" else 0
                        if is_cache_enabled:
                            await cache.set(cache_key, CacheEntry(
                                result=f"[分析失败]",
                                content_hash=content_hash,
                                provider_id=provider_id,
                                model_id=str(model_id),
                                prompt_hash=prompt_hash,
                                analysis_mode="independent",
                                created_at=now,
                                expires_at=now + failure_ttl,
                            ))
                        if can_analyze and failure_mode == "error":
                            # error 模式：当前图片分析失败 → 抛出让上层返回 502
                            raise
                        # skip / fail-open：注入失败说明后继续
                        url_results[url] = f"[视觉分析失败: {e.message}]"
                        url_to_status[url] = "error"
                elif is_historical and historical_cache_miss == "drop":
                    # 静默丢弃：不注入占位文本，注入阶段会直接移除该图片块
                    url_results[url] = ""
                    url_to_status[url] = "dropped"
                    await cache.set(cache_key, CacheEntry(
                        result="",
                        content_hash=content_hash,
                        provider_id=provider_id,
                        model_id=str(model_id),
                        prompt_hash=prompt_hash,
                        analysis_mode="independent",
                        created_at=0,
                        expires_at=0,
                    ))
                elif is_historical and historical_cache_miss == "error":
                    url_results[url] = ""
                    url_to_status[url] = "error"
                    raise VisionAnalysisError(f"Historical image cache miss and historical_cache_miss=error for {url}")
                else:
                    url_results[url] = "[图片分析结果不可用]"
                    url_to_status[url] = "error"

    # Phase 1: resolve all unique URLs（并发）
    await asyncio.gather(*(_process_url(url, img_list) for url, img_list in url_to_images.items()))

    # Phase 2: distribute results to all positions（不覆盖联合分析已设置的位置）
    for url, img_list in url_to_images.items():
        result_text = url_results.get(url, "[未知错误]")
        for img in img_list:
            if img.position not in descriptions:
                descriptions[img.position] = result_text

    # ── Phase 3: 单图增强（长截图切片 OCR / 定位-放大-再读）────────
    # 仅非流式、仅单个当前图片、且该图未被联合分析处理时生效；失败自动保留原结果。
    long_used = False
    grounding_used = False
    structured_used = False
    if stream_queue is None:
        _current_url_set = {img.url for img in images if img.position in allow_analysis_positions}
        current_single_url = next(iter(_current_url_set)) if len(_current_url_set) == 1 else None
        if current_single_url and current_single_url in url_to_raw_bytes and current_single_url not in combined_urls:
            raw = url_to_raw_bytes[current_single_url]
            data_url = url_to_data_url.get(current_single_url, "")
            enhanced_text: str | None = None
            if getattr(cfg.image, "long_screenshot_ocr", False):
                enhanced_text = await _resolve_long_screenshot(
                    raw_bytes=raw,
                    provider_ids=provider_chain,
                    explicit_model=explicit_model,
                    system_prompt=cache_prompt_text if is_cache_enabled else vision_prompt_text,
                    request_client=request_client,
                )
            if enhanced_text is None and getattr(model_config, "grounding_zoom", False) and user_question:
                enhanced_text = await _resolve_grounding_zoom(
                    raw_bytes=raw,
                    data_url=data_url,
                    provider_ids=provider_chain,
                    explicit_model=explicit_model,
                    grounding_prompt=_load_prompt_content("prompts/grounding.txt", user_question),
                    system_prompt=effective_prompt,
                    user_question=user_question,
                    request_client=request_client,
                )
            if enhanced_text:
                long_used = bool(getattr(cfg.image, "long_screenshot_ocr", False) and "分段分析" in enhanced_text)
                grounding_used = "目标元素放大分析" in enhanced_text
                for img in images:
                    if img.position in allow_analysis_positions and img.url == current_single_url:
                        descriptions[img.position] = enhanced_text

    # ── Phase 3.5: 结构化证据格式化（structured_evidence，仅非流式）──
    # 把视觉模型的 JSON 证据转成【摘要】/【全文文字】/【版面结构】等易引用文本，
    # 不确定项单独标注，防止视觉幻觉被源模型当事实采信（参考 modlens 输出契约）。
    if stream_queue is None and getattr(model_config, "structured_evidence", False):
        for img in images:
            pos = img.position
            if pos in descriptions:
                structured = _structure_evidence(descriptions[pos])
                if structured:
                    descriptions[pos] = structured
                    structured_used = True

    if stream_queue is not None:
        await stream_queue.put(("done", "", ""))

    # ── 诊断日志 ────────────────────────────────────────────────
    try:
        status_counts: dict[str, int] = {}
        for s in url_to_status.values():
            status_counts[s] = status_counts.get(s, 0) + 1
        logger.info(
            f"[vision] images={len(images)} statuses={status_counts} "
            f"combined={'combined' if combined_urls else '-'} "
            f"long_screenshot={long_used} grounding={grounding_used} "
            f"structured={structured_used} vision_tokens={vision_usage or '-'}"
        )
    except Exception:
        pass

    return descriptions, vision_usage


async def analyze_images(
    messages: list[dict],
    image_urls: list[str],
    vision_provider_key: str,
    vision_model: str,
    vision_prompt: str,
    request_client: httpx.AsyncClient | None = None,
) -> dict:
    """Legacy single-batch analysis (kept for admin test tool compat)."""
    from app.config import get_config
    cfg = get_config()

    provider_cfg = cfg.vision_providers.get(vision_provider_key)
    if not provider_cfg or not provider_cfg.enabled:
        raise VisionAnalysisError(f"Vision provider '{vision_provider_key}' not found or disabled")

    system_prompt = _load_prompt_content(vision_prompt)
    user_question = extract_user_question(messages)

    resolved_urls = []
    for url in image_urls:
        resolved = await resolve_image_to_base64(url, request_client)
        resolved_urls.append(resolved)

    return await _call_vision_model(
        provider_id=vision_provider_key,
        model_id=vision_model,
        system_prompt=system_prompt,
        user_question=user_question,
        image_urls=resolved_urls,
        request_client=request_client,
    )


def prepare_enhanced_messages(
    messages: list[dict],
    vision_result: str,
) -> list[dict]:
    """Inject vision result into the last user message (legacy compat)."""
    from app.image_utils import remove_image_content as _remove

    def _inject(msgs, injection):
        result = [dict(m) for m in msgs]
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                content = result[i].get("content")
                if isinstance(content, str):
                    result[i]["content"] = f"{content}\n\n{injection}"
                elif isinstance(content, list):
                    new_content = list(content)
                    new_content.append({"type": "text", "text": f"\n\n{injection}"})
                    result[i]["content"] = new_content
                else:
                    result[i]["content"] = injection
                break
        return result

    cleaned = _remove(messages)
    return _inject(cleaned, _build_injection_text(vision_result))
