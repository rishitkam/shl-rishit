FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENABLE_RERANKER=true \
    ENABLE_QUERY_EXPANSION=true

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces run as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user

# Set environment variables for cache directories so models can download successfully
ENV HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface

# Set PATH for the local user bins if needed
ENV PATH="/home/user/.local/bin:$PATH"

# Copy application code with proper ownership
COPY --chown=user:user . .

# Expose the Hugging Face port
EXPOSE 7860

# Start command (Hugging Face passes PORT dynamically, but defaults to 7860)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
