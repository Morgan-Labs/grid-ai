# syntax=docker/dockerfile:1.4

# --- Builder Stage: Install dependencies and build wheels ---
FROM python:3.12.1-slim-bookworm AS builder

WORKDIR /usr/src/app

# 1. Install build tools (for compiling any wheels) with apt cache
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy pyproject and locked requirements to leverage caching
COPY pyproject.toml requirements.txt ./

# 3. Upgrade pip and build wheels for all Python dependencies (cached)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip wheel --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# 4. Copy application source code and build a wheel for the project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-deps --wheel-dir /usr/src/app/wheels .

# --- Final Stage: Set up runtime image ---
FROM python:3.12.1-slim-bookworm

WORKDIR /usr/src/app
# Environment variables (avoid pip cache disable)
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# 5. Install runtime OS dependencies (PDF/OCR libs) with caching, and create app user
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils libmagic1 libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m appuser

# 6. Install Python dependencies from built wheels (enable pip cache) and Gunicorn
COPY --from=builder /usr/src/app/wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /wheels/* gunicorn && \
    # If spaCy is present (via unstructured), download the small English model
    if pip show unstructured | grep -q "spacy"; then \
        python -m spacy download en_core_web_sm --no-deps; \
    fi && \
    # Remove wheel files to clean up image
    rm -rf /wheels

# 7. Set permissions for data and temp directories
RUN mkdir -p /data /tmp/ai_grid_documents && \
    chmod -R 777 /data /tmp && \
    chown -R appuser:appuser /data /tmp/ai_grid_documents /usr/src/app

# Switch to non-root user for safety
USER appuser

# Healthcheck for the service
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ping || exit 1

# Run the application with Gunicorn (single worker for Milvus DB safety)
CMD gunicorn app.main:app -k uvicorn.workers.UvicornWorker --timeout 1800 --keep-alive 120 \
    --workers 1 --bind 0.0.0.0:${PORT:-8000}
