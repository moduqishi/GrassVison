"""协议适配层：把 Anthropic Messages / OpenAI Responses 请求转换为内部 OpenAI 格式，
复用 handle_chat_completion 核心（视觉分析、服务端无感重看、像素注入、缓存），
再把响应转回协议格式（含流式事件）。

设计：
- 入站：协议请求 → OpenAI 格式消息列表（图片提取为 data URL，工具转换，tool_use/tool_result 转换）
- 核心：调用 handle_chat_completion（OpenAI 域）
- 出站：OpenAI 响应 → 协议格式（非流式 JSON / 流式 SSE events）
"""
