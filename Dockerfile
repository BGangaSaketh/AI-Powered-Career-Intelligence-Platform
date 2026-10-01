# ============================================================
# AI-Powered Career Intelligence Platform — Production Dockerfile
# ============================================================

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    STREAMLIT_PORT=8501

# Set working directory
WORKDIR /app

# Install system dependencies (FFmpeg for audio processing, SQLite3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sqlite3 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data required for preprocessing
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application source code
COPY . .

# Create directory structure for runtime storage
RUN mkdir -p uploads temp_audio data

# Expose API and Streamlit ports
EXPOSE 5000 8501

# Production start command (Launches backend API server)
CMD ["python", "app.py"]
