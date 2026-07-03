# Tasks

- [x] Task 1: 修改 `VisionProviderConfig` schema
  - [x] 在 `app/schemas.py` 中将 `disable_thinking: bool = False` 替换为 `extra_params: dict[str, Any] = Field(default_factory=dict)`
  - [x] 导入 `Any` from `typing`（若未导入）
  - [x] 保留 `model_config = {"extra": "ignore"}` 以容忍旧字段

- [x] Task 2: 实现配置加载时的自动迁移
  - [x] 在 `app/config.py` 中找到加载/构造 `VisionProviderConfig` 的位置，添加迁移逻辑：若原始数据含 `disable_thinking` 且不含 `extra_params`，则转换为 `{enable_thinking: false, thinking: false}` 并记录 WARNING 日志
  - [x] 若同时存在 `disable_thinking: true` 与非空 `extra_params`，以 `extra_params` 为准，记录 WARNING
  - [x] 使用 logging 模块（项目已有 logger 时复用）

- [x] Task 3: 修改 `_call_vision_model` payload 构造
  - [x] 在 `app/vision.py` 中移除 `if provider_cfg.disable_thinking:` 块
  - [x] 改为：`payload.update(provider_cfg.extra_params or {})`，位置在 `payload` 初始化之后、`client.post` 之前
  - [x] 注意：`extra_params` 中的 `model`、`messages`、`stream` 等核心字段可被用户覆盖，但这是用户自己的责任（保持简单，不做白名单过滤）

- [x] Task 4: 更新视觉渠道管理前端（键值对动态输入）
  - [x] 在 `templates/vision_providers.html` 中：
    - 表头移除"关闭思考"列；新增"自定义参数"列（展示已配置条数或 `—`）
    - 表单移除"关闭思考"复选框，新增"自定义参数"区块：使用 `x-for` 渲染 `form.extra_params` 数组（每项 `{key:'', value:''}`），每行两个 input（key、value）+ 一个"删除"按钮
    - 区块底部一个"+ 添加参数"按钮，点击 push 一个空 `{key:'', value:''}` 到数组
    - `openCreate` 中 `form.extra_params = []`
    - `editProvider(key, p)` 时把 `p.extra_params`（dict）转为 `[{key:k, value:String(v)}, ...]` 数组赋给 `form.extra_params`
    - `saveProvider` 提交前：
      - 过滤掉 key 与 value 同时为空的行
      - 若某行只有 key 没 value（或反之），`showToast('自定义参数的 key 和 value 必须同时填写','error')` 并 return
      - 把数组转回 dict：`Object.fromEntries(arr.map(r => [r.key, parseValue(r.value)]))`，其中 `parseValue` 尝试 JSON.parse 识别 `true/false/null/数字`，失败则返回原字符串
      - 提交的 body 中 `extra_params` 为该 dict
  - [x] 列表展示列：把原"关闭思考"列改为"自定义参数"，使用 `<code x-text="Object.keys(p.extra_params||{}).length ? Object.keys(p.extra_params).join(', ') : '—'"></code>` 简洁展示已配置的 key 列表

- [x] Task 5: 修复 settings 保存后不刷新
  - [x] 在 `templates/settings.html` 的 `saveSettings()` 中：保存成功后 `showToast('已保存')` 然后 `setTimeout(() => location.reload(), 600)`（给 toast 留时间显示）
  - [x] 失败时不刷新（保留输入）；`needsRestart` 逻辑保留作为重启提示
  - [x] 注意：`needs_restart` 仍由后端返回，reload 后页面通过初始上下文不再显示 needsRestart（重启仍需用户手动操作）

- [x] Task 6: 更新示例配置
  - [x] 在 `config.example.yaml` 中将 `disable_thinking: false` 注释行替换为 `extra_params: {}` 并附注释说明
  - [x] 在注释中给出 Qwen-VL / DeepSeek / Claude 等模型的关闭思考参数示例

- [x] Task 7: 补充测试
  - [x] 在 `tests/test_config.py` 中新增测试：
    - 旧配置 `disable_thinking: true` 加载后 `extra_params == {enable_thinking: false, thinking: false}`
    - 旧配置 `disable_thinking: false` 加载后 `extra_params == {}`
    - 同时存在 `disable_thinking: true` 与 `extra_params` 时以 `extra_params` 为准
    - 仅 `extra_params` 配置加载后保持原值
  - [x] 运行 `pytest tests/test_config.py` 通过

# Task Dependencies
- Task 2 依赖 Task 1（schema 字段定义）
- Task 3 依赖 Task 1
- Task 4 依赖 Task 1
- Task 7 依赖 Task 1、Task 2
- Task 5、Task 6 独立可并行
