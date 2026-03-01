# Multi-stage Dockerfile for Flight Delay Prediction Platform

# Stage 1: Build dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production server
FROM python:3.10-slim

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Make sure scripts in .local are usable
ENV PATH=/home/app/.local/bin:$PATH

# Copy application code
COPY . .

# Download pre-trained model binaries from GitHub
# This bypasses Hugging Face's strict binary Git tracking blocks
RUN python -c "import os, urllib.request; \
    os.makedirs('models', exist_ok=True); \
    os.makedirs('data/processed', exist_ok=True); \
    urllib.request.urlretrieve('https://raw.githubusercontent.com/faiz0304/Flight-Delays-Predictor/main/models/classification_model.pkl', 'models/classification_model.pkl'); \
    urllib.request.urlretrieve('https://raw.githubusercontent.com/faiz0304/Flight-Delays-Predictor/main/models/regression_model.pkl', 'models/regression_model.pkl'); \
    urllib.request.urlretrieve('https://raw.githubusercontent.com/faiz0304/Flight-Delays-Predictor/main/data/processed/label_encoders.pkl', 'data/processed/label_encoders.pkl')"

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port for FastAPI
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]