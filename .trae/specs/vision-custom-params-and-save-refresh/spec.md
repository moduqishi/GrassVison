# 视觉渠道自定义参数 & 设置保存刷新 Spec

## Why
当前视觉渠道的"关闭思考"选项以布尔值硬编码为发送 `enable_thinking=false` 与 `thinking=false` 两个参数，但不同模型关闭思考/推理模式所需的参数名和形式各不相同（如 Qwen-VL 用 `enable_thinking`、部分模型用 `thinking`、Claude 用 `thinking.type=disabled`、DeepSeek 用 `reasoning_effort=none` 等），无法满足多模型场景。同时，系统设置页保存成功后页面不会自动刷新，导致密码字段残留、计数显示滞后等问题。

## What Changes
- **BREAKING**: 移除 `VisionProviderConfig.disable_thinking: bool` 字段，新增 `extra_params: dict[str, Any]` 字段，用于保存任意外部请求参数（合并到 vision 模型请求 payload 中）
- 修改 `app/vision.py` 的 `_call_vision_model`：不再判断 `disable_thinking`，改为将 `extra_params` 浅合并到请求 payload 中（用户自定义参数优先级高于默认）
- 修改 `templates/vision_providers.html`：移除"关闭思考"复选框与表格列，改为键值对动态输入（每行一个 key/value 输入框，可增删行）；展示列也相应调整
- 修改 `app/schemas.py`：字段替换
- 修改 `config.example.yaml`：更新示例配置并提供常见模型关闭思考的参数示例
- 修复 `templates/settings.html` 的 `saveSettings()`：保存成功后调用 `location.reload()` 刷新页面
- 数据迁移：旧配置中的 `disable_thinking: true` 自动迁移为 `extra_params: {enable_thinking: false, thinking: false}`，避免用户配置丢失

## Impact
- Affected specs: 无（首次 spec）
- Affected code:
  - [app/schemas.py](file:///Users/cake/toys/GrassVison/app/schemas.py) — `VisionProviderConfig` 字段
  - [app/vision.py](file:///Users/cake/toys/GrassVison/app/vision.py) — `_call_vision_model` payload 构造
  - [app/admin.py](file:///Users/cake/toys/GrassVison/app/admin.py) — vision provider 创建/更新接口（接收新字段，处理迁移）
  - [app/config.py](file:///Users/cake/toys/GrassVison/app/config.py) — 配置加载时迁移 `disable_thinking` → `extra_params`
  - [templates/vision_providers.html](file:///Users/cake/toys/GrassVison/templates/vision_providers.html) — 表单与表格
  - [templates/settings.html](file:///Users/cake/toys/GrassVison/templates/settings.html) — 保存后刷新
  - [config.example.yaml](file:///Users/cake/toys/GrassVison/config.example.yaml) — 示例配置
  - [tests/test_config.py](file:///Users/cake/toys/GrassVison/tests/test_config.py) — 迁移逻辑测试

## ADDED Requirements

### Requirement: 自定义请求参数
视觉渠道 SHALL 支持通过 `extra_params` 字段配置任意外部请求参数，这些参数会在调用视觉模型 `/chat/completions` 时被合并到请求 payload 中。

#### Scenario: 用户配置自定义参数
- **WHEN** 用户在视觉渠道表单的"自定义参数"区块点击"+ 添加参数"
- **AND** 在新增行中填写 key（如 `enable_thinking`）和 value（如 `false`）
- **AND** 触发一次视觉分析
- **THEN** 系统将所有键值对组装为 dict 合并到发送给视觉模型的 payload 中
- **AND** 用户自定义参数会覆盖默认的同名字段（除 `model`、`messages` 等核心字段外的次要参数）

#### Scenario: 留空时不影响请求
- **WHEN** 用户未添加任何自定义参数行（或所有行均为空）
- **THEN** 视觉模型请求 payload 与默认行为完全一致，不附加任何额外参数

#### Scenario: 动态增删参数行
- **WHEN** 用户点击"+ 添加参数"按钮
- **THEN** 在参数列表末尾追加一行空的 key/value 输入框
- **WHEN** 用户点击某行的"删除"按钮
- **THEN** 该行被移除
- **WHEN** 用户编辑现有的视觉渠道
- **THEN** 已保存的 `extra_params` 中的每个键值对渲染为对应的输入行

#### Scenario: 表单输入校验
- **WHEN** 用户填写了 key 但 value 为空（或反之）
- **THEN** 前端阻止提交并提示"自定义参数的 key 和 value 必须同时填写"
- **WHEN** 用户输入的 value 无法被解析为数字/布尔/null
- **THEN** 该值作为字符串发送（不做强制类型转换，但允许 `true`/`false`/`null`/数字 的自然识别）

### Requirement: 设置保存后自动刷新
系统设置页保存成功后 SHALL 自动刷新页面，确保所有显示字段（含敏感字段与计数）与最新后端状态一致。

#### Scenario: 保存设置成功
- **WHEN** 用户点击"保存设置"按钮
- **AND** 后端返回成功
- **THEN** 显示"已保存"提示后页面自动 reload
- **AND** 密码字段不再残留用户输入

#### Scenario: 保存设置失败
- **WHEN** 用户点击"保存设置"按钮
- **AND** 后端返回错误
- **THEN** 显示错误提示，页面不刷新（保留用户输入便于修正）

## MODIFIED Requirements

### Requirement: 视觉渠道配置模型
`VisionProviderConfig` 包含以下字段：`name`、`enabled`、`base_url`、`api_key`、`model`、`timeout`、`max_images`、`max_image_size_mb`、`extra_params`（dict，默认 `{}`）、`headers`。原 `disable_thinking` 字段移除。

#### Scenario: 加载旧配置自动迁移
- **WHEN** 配置文件中存在旧的 `disable_thinking: true` 字段
- **THEN** 系统加载时自动转换为 `extra_params: {enable_thinking: false, thinking: false}`
- **AND** 日志记录一条 WARNING 提示用户迁移至 `extra_params`
- **WHEN** 配置文件中存在旧的 `disable_thinking: false` 字段
- **THEN** 系统加载时忽略该字段，`extra_params` 保持默认 `{}`

#### Scenario: 已有 `extra_params` 时不被旧字段覆盖
- **WHEN** 配置文件同时存在 `disable_thinking: true` 与 `extra_params: {reasoning_effort: none}`
- **THEN** 以 `extra_params` 为准，不进行合并迁移
- **AND** 日志记录 WARNING 提示冲突

## REMOVED Requirements

### Requirement: 关闭思考开关
**Reason**: 不同模型关闭思考的参数名与形式不一致，硬编码 `enable_thinking=false` + `thinking=false` 无法覆盖主流模型，改为通用的 `extra_params` 自定义参数方案。
**Migration**: 配置加载时自动迁移：`disable_thinking: true` → `extra_params: {enable_thinking: false, thinking: false}`。用户需在 UI 上重新审视并按目标模型文档调整 `extra_params` 内容（例如 Qwen-VL 用 `{"enable_thinking": false}`、DeepSeek-R1 用 `{"reasoning_effort": "none"}` 等）。
