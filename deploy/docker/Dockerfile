# AIPROD API - Production Dockerfile
# PHASE 4 - GCP Production Hardening

FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    ffprobe \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY packages/aiprod-pipelines/pyproject.toml /app/
COPY packages/aiprod-pipelines/src /app/src
COPY packages/aiprod-core/src /app/aiprod_core

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    google-cloud-storage==2.10.0 \
    google-cloud-logging==3.8.0 \
    google-cloud-monitoring==2.16.0 \
    google-generativeai==0.3.0 \
    pydantic==2.5.0 \
    cachetools==5.3.2 \
    aiohttp==3.9.1 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1

# Copy application code
COPY packages/aiprod-pipelines/src/aiprod_pipelines /app/aiprod_pipelines

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["uvicorn", "aiprod_pipelines.api.endpoints:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
