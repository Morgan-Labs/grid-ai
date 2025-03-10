# syntax=docker/dockerfile:1.4
# Enable BuildKit features

# Build stage
FROM python:3.12.1-slim-bookworm AS builder

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies - only what's needed for building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install specific pydantic versions first (as specified in pyproject.toml)
RUN pip install --no-cache-dir pydantic==2.10.6 pydantic-core==2.27.2
RUN pip install --upgrade pip setuptools wheel

# Copy only the requirements-related files first for better caching
COPY pyproject.toml ./

# Install python dependencies to generate wheels
RUN pip wheel --no-deps --wheel-dir /usr/src/app/wheels -r <(pip install tomli && python -c "import tomli; f = open('pyproject.toml', 'rb'); data = tomli.load(f); print('\n'.join(data['project']['dependencies']))") && \
    pip wheel --no-deps --wheel-dir /usr/src/app/wheels 'pymupdf>=1.23.0' 'pypdf>=3.15.1' 'unstructured[pdf]' 'pdf2image>=1.16.3' 'pytesseract>=0.3.10' 'boto3>=1.34.0' 'amazon-textract-caller>=0.2.0' 'langchain-openai>=0.2.0' 'openai>=1.45.1' 'pillow>=10.0.0'

# Final stage
FROM python:3.12.1-slim-bookworm

# Create a non-root user
RUN useradd -m appuser

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy wheels from builder stage
COPY --from=builder /usr/src/app/wheels /wheels
COPY pyproject.toml ./

# Install system dependencies for PDF processing
# All in a single RUN command to avoid lock issues
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy source code (which changes most frequently) last for better caching
COPY src ./src

# Create data directory and set permissions
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
