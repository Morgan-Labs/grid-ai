# syntax=docker/dockerfile:1.4
# Enable BuildKit features

##############################
# STEP 1: Dependencies Layer
##############################
FROM python:3.12.1-slim-bookworm AS dependencies

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system build dependencies - these are only needed for building packages
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# IMPORTANT: Copy only pyproject.toml for optimal caching
# This layer will remain cached unless pyproject.toml changes
COPY pyproject.toml ./

# Install dependencies with specific pydantic versions
# This will be cached as long as pyproject.toml doesn't change
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    # Using the version constraints from pyproject.toml
    pip install -e ".[pdf-all]" && \
    # Only install spaCy model if needed by unstructured
    if pip show unstructured | grep -q "spacy"; then \
        python -m spacy download en_core_web_sm --no-deps; \
    fi

##############################
# STEP 2: Final Image
##############################
FROM python:3.12.1-slim-bookworm

# Create a non-root user
RUN useradd -m appuser

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install ONLY system runtime dependencies
# These are needed for running the app, not building packages
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from dependencies stage
# This avoids reinstalling packages with every build
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy pyproject.toml for package metadata
COPY pyproject.toml ./

# Copy source code (which changes most frequently) last for better caching
# This is the step that will run on most code changes
COPY src ./src

# Create data directory with correct permissions
RUN mkdir -p /data && chown -R appuser:appuser /data && chmod 777 /data && \
    chown -R appuser:appuser /usr/src/app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ping || exit 1

# Run the application
# Use PORT environment variable if set (for Render), otherwise default to 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
