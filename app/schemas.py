"""Pydantic models for configuration and OpenAI-compatible request/response."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class SourceProviderConfig(BaseModel):
    model_config = {"extra": "ignore"}
    name: str = ""
    enabled: bool = True
    base_url: str = ""
    api_key: str = ""
    timeout: int = 120
    headers: dict[str, str] = Field(default_factory=dict)


class VisionProviderConfig(BaseModel):
    model_config = {"extra": "ignore"}
    name: str = ""
    enabled: bool = True
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 120
    max_images: int = 5
    max_image_size_mb: int = 10
    max_tokens: int = 4096  # 视觉模型输出上限（长文档/长代码分析可调大）
    image_detail: str = ""  # 发送给视觉模型的 image_url.detail 值: "" | "auto" | "low" | "high"; 空 = 不发送该字段
    extra_params: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class EnhancedModelConfig(BaseModel):
    model_config = {"extra": "ignore"}
    name: str = ""
    enabled: bool = True
    source_provider: str = ""
    source_model: str = ""
    vision_enabled: bool = False
    vision_provider: str = ""
    vision_model: str = ""
    vision_prompt: str = "prompts/default.txt"
    vision_failure_mode: Literal["error", "skip"] = "error"
    replace_response_model: bool = True
    cache_prompt: str | None = None
    # 视觉渠道故障转移链：主渠道失败时按序尝试这些渠道（渠道 key 列表）
    vision_provider_failover: list[str] = Field(default_factory=list)
    # 定位-放大-再读：用户问题针对具体 UI 元素时，先定位坐标框，再裁剪放大做二次精读
    grounding_zoom: bool = False
    # 协议化服务端重看：注入 grassvision_view_image 工具，源模型描述不足时自主调用，
    # GrassVision 在服务端用请求内缓存的图片字节重新分析（无需用户重发），客户端无感知
    vision_reexamine: bool = False
    # 结构化证据输出：视觉模型返回 JSON 证据（摘要/全文/版面/实体/不确定项），
    # 解析校验后格式化为易引用文本注入（不确定项单独标注）
    structured_evidence: bool = False


class ServerConfig(BaseModel):
    model_config = {"extra": "ignore"}
    host: str = "127.0.0.1"
    port: int = 8042
    access_key: str = ""
    request_timeout: int = 180


class AdminConfig(BaseModel):
    model_config = {"extra": "ignore"}
    enabled: bool = True
    username: str = "admin"
    password: str = ""


class VisionCacheConfig(BaseModel):
    model_config = {"extra": "ignore"}
    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 200
    default_prompt: str = "prompts/cache.txt"


class ImageConfig(BaseModel):
    model_config = {"extra": "ignore"}
    max_images: int = 5
    max_image_size_mb: int = 10
    max_total_size_mb: int = 30
    max_width: int = 4096
    max_height: int = 4096
    download_timeout: int = 20
    allow_private_network: bool = False
    multi_image_mode: str = "independent"  # independent=逐图独立分析 | auto=对比意图时联合 | combined=多图总是联合一次调用
    analysis_scope: str = "latest_user_message"
    historical_cache_miss: Literal["analyze", "drop", "error"] = "analyze"
    comparison_strategy: str = "source_model"
    thinking_guidance: bool = False  # 注入系统提示，引导源模型在思考链中引用图片分析
    stream_vision_thinking: bool = False  # 流式透传视觉模型的思考/分析过程，再无缝衔接源模型
    # 流式视觉阶段是否推送"正在处理图片"预提示：默认关闭，追求真实思考链（首帧即视觉模型真实输出）
    vision_stream_prelude: bool = False
    # 通道说明注入：告诉源模型"收到的是文字分析不是像素"，细节不足时主动引导用户重发图片
    # 从而触发带新意图的重新分析（模拟原生多模态的按需重看，参考 agent-vision-toolkit）
    vision_channel_note: bool = False
    # 问题感知缓存：开启后把用户问题格式化进视觉 prompt，缓存键随问题变化，
    # 同一图片不同问题分别分析（针对性更强、缓存命中率下降）
    question_aware_cache: bool = False
    # 历史轮次图片：开启后缓存命中的历史图片描述原地注入（不触发新分析），
    # 保住多轮追问上下文；未命中时按 historical_cache_miss 处理
    reuse_historical_cache: bool = False
    # 多图并发分析的并发度上限
    vision_concurrency: int = 4
    # 长截图切片 OCR：高宽比≥3 且高度≥1800px 的截图自动分段分析后合并
    long_screenshot_ocr: bool = False
    vision_cache: VisionCacheConfig = Field(default_factory=VisionCacheConfig)


class LoggingConfig(BaseModel):
    model_config = {"extra": "ignore"}
    level: str = "INFO"
    save_to_file: bool = True
    file: str = "logs/grassvision.log"
    log_request_body: bool = False
    log_vision_result: bool = False


class AppConfig(BaseModel):
    model_config = {"extra": "ignore"}
    server: ServerConfig = Field(default_factory=ServerConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    source_providers: dict[str, SourceProviderConfig] = Field(default_factory=dict)
    vision_providers: dict[str, VisionProviderConfig] = Field(default_factory=dict)
    models: dict[str, EnhancedModelConfig] = Field(default_factory=dict)
    image: ImageConfig = Field(default_factory=ImageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    stop: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    n: int | None = 1
    user: str | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "grassvision"
