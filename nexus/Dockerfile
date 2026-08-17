FROM python:3.11-slim

WORKDIR /app

COPY nexus/requirements.txt ./nexus/requirements.txt
RUN pip install --no-cache-dir -r nexus/requirements.txt

COPY nexus ./nexus

ENV PYTHONUNBUFFERED=1 \
    NEXUS_ENV=production

EXPOSE 8100

CMD ["uvicorn", "nexus.api.main:app", "--host", "0.0.0.0", "--port", "8100"]
