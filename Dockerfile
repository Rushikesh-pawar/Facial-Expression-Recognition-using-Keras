# CPU-only container for the inference web app.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# OpenCV + MediaPipe runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY templates ./templates
COPY static ./static

EXPOSE 5000

ENV MODEL_PATH=/app/artifacts/models/emotion_classifier_final.keras
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "app.app:create_app()"]
