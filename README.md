# 🌿 GrassVision

给 DeepSeek 等纯文本大模型外挂图像理解能力的本地中转服务。提供 OpenAI 兼容的 API，自动将图片请求交给视觉模型分析，再将结构化结果注入文本模型，使增强后的模型体验接近原生多模态。

## 快速开始

```bash
# 1. 安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入实际的 API Key

# 3. 启动服务
uvicorn app.main:app --host 127.0.0.1 --port 8042
```

## 使用方式

把客户端的 API Base URL 改为 GrassVision：

```
http://127.0.0.1:8042/v1
```

### 支持的图片格式

- HTTP(S) URL 图片
- Base64 Data URL 图片 (`data:image/png;base64,...`)
- 单张和多张图片

### 增强模型

默认内置 `deepseek-vision` 模型：源模型为 DeepSeek Chat，视觉模型为 GLM-4V-Flash。

可通过管理界面 `/admin` 创建更多增强模型：

- `deepseek-ocr` — 强制 OCR 模式提取文字
- `deepseek-code-vision` — 代码/错误诊断
- `deepseek-ui-vision` — UI 截图分析

### 自动图片类型检测

默认 `deepseek-vision` 使用智能 prompt，能自动判断图片类型（代码截图/技术图表/数据图表/UI/文档/通用图片）并按对应策略分析。

## 管理界面

```
http://127.0.0.1:8042/admin
```

默认用户名 `admin`，密码在 `.env` 中配置。

功能：
- 源渠道 CRUD + 连接测试
- 视觉渠道 CRUD + 图片分析测试
- 增强模型 CRUD + 关联保护
- 视觉提示词管理
- 在线测试（选模型、发图片、看调试信息）
- 系统设置（服务/图片限制/日志）
- 配置预览 + 手动编辑 YAML
- 日志查看 + 搜索筛选

## 配置

主配置文件 `config/config.yaml`，支持 `${ENV_VAR}` 引用 `.env`。

保存配置时自动备份到 `config/backups/`，保留最近 10 份。使用原子写入，校验失败不破坏原配置。

## Docker

```bash
docker-compose up -d
```

## 项目结构

```
GrassVision/
├── app/              # FastAPI 应用
│   ├── main.py       # 入口 + 路由
│   ├── config.py     # 配置加载/重载/原子写入
│   ├── schemas.py    # Pydantic 模型
│   ├── proxy.py      # 核心代理（检测/路由/流式）
│   ├── vision.py     # 视觉分析 + 结果注入
│   ├── image_utils.py# 图片提取/下载/校验
│   ├── providers.py  # HTTPX 客户端管理
│   ├── auth.py       # 管理界面 Session 认证
│   └── admin.py      # 管理 API CRUD
├── templates/        # Jinja2 管理界面
├── static/css/       # 原生 CSS
├── config/           # YAML 配置 + prompts + 备份
├── tests/            # 测试
└── logs/             # 日志文件
```
