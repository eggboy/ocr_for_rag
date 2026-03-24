###############################################################################
# Stage 1 – Install Python dependencies
###############################################################################
FROM python:3.12.11-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /tmp/build

RUN uv init --no-readme --no-pin-python && \
    uv add \
        "huggingface_hub[cli]" \
        "transformers>=5.3.0" \
        "glmocr[server,selfhosted]"

###############################################################################
# Stage 2 – Download model weights (cacheable independently of code changes)
###############################################################################
FROM deps AS model-fetch

RUN uv run hf download zai-org/GLM-OCR \
        --local-dir /model/GLM-OCR

###############################################################################
# Stage 3 – Final runtime image
###############################################################################
FROM python:3.12.11-slim

LABEL org.opencontainers.image.source="https://github.com/eggboy/ocr_for_rag" \
      org.opencontainers.image.description="GLM-OCR self-hosted server"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/GLM-OCR

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --no-create-home appuser

COPY --from=deps   /opt/venv       /opt/venv
COPY --from=model-fetch /model/GLM-OCR /app/GLM-OCR

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

ENTRYPOINT ["python", "-m", "glmocr.server"]
CMD ["--host", "0.0.0.0", "--port", "5000", "--layout-device", "cpu", "--glm-device", "cpu"]
