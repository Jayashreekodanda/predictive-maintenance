FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps first (important for numpy/pandas/tensorflow performance)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install heavy packages separately with retries and long timeout
RUN pip install --no-cache-dir --timeout=1000 --retries=10 \
    tensorflow==2.19.0 \
    xgboost==3.0.4 \
    scikit-learn==1.6.1

# Install remaining smaller dependencies
RUN pip install --no-cache-dir --timeout=1000 --retries=10 \
    numpy==2.0.2 pandas==2.2.2 joblib==1.3.2 psutil==5.9.8

# Copy project files
COPY models/ models/
COPY data/   data/
COPY scripts/ scripts/

CMD ["python", "scripts/benchmark.py", "rf"]
