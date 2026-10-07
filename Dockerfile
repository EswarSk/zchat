FROM python:3.11.15-slim
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/models/huggingface XDG_CACHE_HOME=/models/cache \
    MODEL_CACHE_DIR=/models/fastembed DATABASE_PATH=/data/metadata.sqlite3
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv==0.12.23
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
RUN useradd --create-home --uid 10001 zchat && mkdir -p /data /models \
    && chown -R zchat:zchat /data /models
USER zchat
EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
