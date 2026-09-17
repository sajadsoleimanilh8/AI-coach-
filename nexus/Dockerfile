# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS build
# Generous timeouts/retries: a dropped connection mid-wheel otherwise fails the
# whole build with a hash mismatch.
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY nexus/requirements.txt /tmp/req/requirements.txt
RUN pip install -r /tmp/req/requirements.txt

FROM python:3.11-slim-bookworm
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    NEXUS_ENV=production

WORKDIR /app
COPY nexus ./nexus
# nexus/sports/game_plan.py imports ai.opponent_intelligence.*, so the image
# cannot start without it. Two pure-Python modules, no extra dependencies.
COPY ai ./ai

# Unprivileged user; nexus/data holds the SQLite memory store and workspace.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/nexus/data \
    && chown -R app:app /app/nexus/data
USER app

EXPOSE 8100

# /api/health is exempt from API-key auth so this probe needs no secret.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "nexus.api.main:app", "--host", "0.0.0.0", "--port", "8100"]
