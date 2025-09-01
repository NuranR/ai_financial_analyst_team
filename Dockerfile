# Use Python 3.10 as base image
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

EXPOSE 8501

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=/app/mlruns

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]