# GPU inference worker. Build from the repo root:
#   docker build -f deployment/docker/inference.Dockerfile -t sports-strategy-inference .
# See deployment/docs.md for the GPU host setup this file cannot automate.

# Base image already carries a torch built against this CUDA; do not pip
# install torch on top of it or a CPU-only wheel can win.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# opencv dlopens libGL at import time even headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
COPY deployment/docker/inference-requirements.txt /app/deployment/docker/inference-requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/docker/inference-requirements.txt

# Fail the build rather than ship an image whose mediapipe lacks the legacy
# Solutions API that pose estimation needs.
RUN python -c "\
import mediapipe as mp; \
assert hasattr(mp.solutions, 'pose'), \
    'mediapipe.solutions.pose is missing from this mediapipe build -- pin a ' \
    'different mediapipe version in inference-requirements.txt (see ' \
    'ai/computer_vision/pose_estimation/pose.py _load_legacy_pose_solution()).'"

COPY backend /app/backend
COPY ai /app/ai

ENV PYTHONPATH=/app

# --pool=solo: CUDA contexts do not survive Celery's prefork.
CMD ["celery", "-A", "backend.celery_app:celery_app", "worker", "--loglevel=info", "--pool=solo"]
