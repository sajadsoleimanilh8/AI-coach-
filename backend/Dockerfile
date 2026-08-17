FROM python:3.11-slim

WORKDIR /app

# opencv dlopens libGL at import time even headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# requirements-full.txt, not requirements.txt: backend.api.main imports the
# ai.* packages at module scope and needs numpy/opencv/scipy/mediapipe/ultralytics.
COPY backend/requirements-full.txt /app/backend/requirements-full.txt
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements-full.txt

COPY backend /app/backend
COPY ai /app/ai

# *.pt checkpoints are gitignored and never baked into this image; mount them
# at runtime (see docker-compose.yml).

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
