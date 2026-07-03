# Checklist

- [x] `app/schemas.py` 中 `VisionProviderConfig` 已用 `extra_params: dict[str, Any]` 替换 `disable_thinking: bool`
- [x] `app/config.py` 实现旧字段 `disable_thinking` 到 `extra_params` 的自动迁移，并记录 WARNING 日志
- [x] `app/vision.py` 中 `_call_vision_model` 使用 `payload.update(provider_cfg.extra_params or {})` 合并自定义参数
- [x] `templates/vision_providers.html` 表头与表单已更新为"自定义参数"，使用键值对动态输入（每行 key+value 输入框 + 删除按钮，底部"+ 添加参数"按钮）
- [x] `templates/vision_providers.html` 中 `editProvider` / `openCreate` / `saveProvider` 正确处理 `extra_params`（dict ↔ 行数组互转，含校验）
- [x] `templates/vision_providers.html` 保存时 value 自动识别 `true/false/null/数字`，否则作为字符串
- [x] `templates/settings.html` 的 `saveSettings()` 成功后调用 `location.reload()`
- [x] `templates/settings.html` 保存失败时不清空输入、不刷新
- [x] `config.example.yaml` 已移除 `disable_thinking`，新增 `extra_params` 与多模型示例注释
- [x] `tests/test_config.py` 新增迁移逻辑测试且全部通过
- [ ] 手动验证：旧 config.yaml（含 `disable_thinking: true`）启动后能正确迁移，vision 调用请求 payload 包含 `enable_thinking=false` 与 `thinking=false`
- [ ] 手动验证：在 UI 中添加一行 `enable_thinking` = `false` 保存后，触发视觉分析时请求 payload 含该参数
- [ ] 手动验证：填写只有 key 没 value（或反之）时前端阻止提交并提示
- [ ] 手动验证：系统设置页保存后页面自动刷新，密码框被清空
