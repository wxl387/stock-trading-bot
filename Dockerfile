FROM python:3.9-slim

# System dependencies for numpy, scikit-learn, xgboost, lightgbm, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

# Create directories for runtime data (will be overridden by volume mounts)
RUN mkdir -p data logs models

# Healthcheck: verify the python process is running
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD pgrep -f "start_trading.py" > /dev/null || exit 1

ENTRYPOINT ["python", "scripts/start_trading.py"]
CMD ["--simulated", "--interval", "120", "--ensemble"]
