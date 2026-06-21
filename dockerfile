FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Explicitly copy artifacts (overrides .dockerignore exclusion)
COPY artifacts/preprocessing.pkl /app/artifacts/preprocessing.pkl
COPY artifacts/model.pkl /app/artifacts/model.pkl
COPY mlp_classification.h5 /app/mlp_classification.h5

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]