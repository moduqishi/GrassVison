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
    multi_image_mode: str = "independent"
    analysis_scope: str = "latest_user_message"
    historical_cache_miss: Literal["analyze", "drop", "error"] = "analyze"
    comparison_strategy: str = "source_model"
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
