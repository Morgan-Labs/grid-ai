# syntax=docker/dockerfile:1.4

# --- Builder Stage: build wheels for dependencies + your app ---
FROM python:3.12.1-slim-bookworm AS builder
WORKDIR /usr/src/app

# 1) Install build-essential (no cache mounts; guaranteed lock-free)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2) Copy lockfiles
COPY pyproject.toml requirements.txt ./

# 3) Build wheels for all Python deps, caching pip
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip wheel --no-deps --wheel-dir=./wheels -r requirements.txt

# 4) Copy your source code and build your app’s wheel
COPY src ./src
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip wheel --no-deps --wheel-dir=./wheels ./src

# --- Final Stage: runtime image ---
FROM python:3.12.1-slim-bookworm
WORKDIR /usr/src/app

# 5) Env for pip caching & quiet Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 6) Install OS-level PDF/OCR libs (fresh, lock-free)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        poppler-utils libmagic1 libgl1 libglib2.0-0 \
        tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# 7) Install all wheels + Gunicorn, with pip caching
COPY --from=builder /usr/src/app/wheels /wheels
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install /wheels/* gunicorn && \
    if pip show unstructured >/dev/null 2>&1; then \
        python -m spacy download en_core_web_sm --no-deps; \
    fi && \
    rm -rf /wheels

# 8) Create dirs, set perms, switch to non-root user
RUN mkdir -p /data /tmp/ai_grid_documents && \
    chmod -R 777 /data /tmp && \
    useradd -m appuser && \
    chown -R appuser:appuser /data /tmp/ai_grid_documents /usr/src/app
USER appuser

# 9) Healthcheck & run
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ping || exit 1

CMD ["gunicorn","app.main:app","-k","uvicorn.workers.UvicornWorker",\
        "--timeout","1800","--keep-alive","120","--workers","1","--bind","0.0.0.0:8000"]
