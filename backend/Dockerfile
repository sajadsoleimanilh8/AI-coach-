# syntax=docker/dockerfile:1

# ---- build stage: resolve and install dependencies into a venv ------------
FROM python:3.11-slim-bookworm AS build

# Generous timeouts/retries: a dropped connection mid-wheel otherwise fails
# the whole build with a hash mismatch.
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU torch, installed explicitly and FIRST. The torch pins in
# requirements-full.txt are Windows/CUDA-only (platform markers), so on Linux
# ultralytics would otherwise resolve the newest CUDA build of torch: several GB
# of nvidia-* wheels in an image that has no GPU. GPU inference has its own
# image: deployment/docker/inference.Dockerfile.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# requirements-full.txt, not requirements.txt: backend.api.main imports the
# ai.* packages at module scope and needs numpy/opencv/scipy/mediapipe/ultralytics.
COPY backend/requirements.txt backend/requirements-full.txt /tmp/req/
RUN pip install -r /tmp/req/requirements-full.txt

# ---- runtime stage ----------------------------------------------------------
FROM python:3.11-slim-bookworm

# opencv dlopens libGL at import time even headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY backend /app/backend
COPY ai /app/ai
# backend/pipeline imports `from configs import registry`; without this the
# image builds fine and fails on the first processing job.
COPY configs /app/configs

# Run as an unprivileged user. storage/ and calibrations/ are written at
# runtime, so the user owns those mount points.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/storage /app/calibrations \
    && chown -R app:app /app/storage /app/calibrations
USER app

# *.pt checkpoints are never baked into this image; mount them at runtime
# (see docker-compose.yml).

EXPOSE 8000

# /health is exempt from API-key auth precisely so this probe needs no secret.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
