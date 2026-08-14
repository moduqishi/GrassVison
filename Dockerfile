FROM python:3.11-slim

WORKDIR /app

# 安装 chromium（供 grassvision_ui_diff 的 HTML 渲染）
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/chromium /usr/bin/chromium-browser 2>/dev/null || true

ENV GRASSVISION_CHROME=/usr/bin/chromium

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p config/backups config/prompts logs

EXPOSE 8042

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8042"]
