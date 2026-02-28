# Multi-stage build for optimized production image with GPU support
# Using CUDA 12.1 with cuDNN 8 and Python 3.11
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

# Install Python 3.11 and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    gcc \
    g++ \   
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies from app directory
COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip install --no-cache-dir --upgrade pip && \
    # Install PyTorch with CUDA 12.1 support first (latest stable versions)
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    # Install other requirements except torch packages (already installed with CUDA support)
    grep -vE "^(torch|torchvision|torchaudio)" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    # Download NLTK data needed for g2p_en (both tagger variants and cmudict)
    python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', download_dir='/opt/nltk_data', quiet=True); nltk.download('averaged_perceptron_tagger', download_dir='/opt/nltk_data', quiet=True); nltk.download('cmudict', download_dir='/opt/nltk_data', quiet=True)" && \
    # Set permissions for nltk_data
    chmod -R 755 /opt/nltk_data && \
    # Remove unnecessary files to reduce size
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "*.dist-info" -exec rm -rf {}/RECORD {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    rm -rf /opt/venv/lib/python*/site-packages/pip /opt/venv/lib/python*/site-packages/setuptools

# Production stage with GPU runtime
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11 and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /workspace

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy NLTK data from builder
COPY --from=builder /opt/nltk_data /opt/nltk_data

# Copy only the app directory from the repository
COPY --chown=appuser:appuser app/ ./app/

# Create cache directories with proper permissions
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch && \
    chown -R appuser:appuser /workspace/.cache && \
    # Remove any unnecessary files that might have been copied
    find ./app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find ./app -type f -name "*.pyc" -delete && \
    find ./app -type f -name "*.pyo" -delete

# Switch to non-root user
USER appuser

# Set minimal environment variables (Cloud Run will inject the rest)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    NLTK_DATA=/opt/nltk_data

# Expose port (Cloud Run will use PORT env var, default to 8080)
ENV PORT=8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8080\")}/api/model/health')" || exit 1

# Run application with gunicorn (optimized for GPU workload)
# Using 2 workers with multiple threads for concurrent requests
# Cloud Run will set PORT environment variable
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 16 --worker-class gthread --timeout 300 --log-level info --access-logfile - --error-logfile - --capture-output --enable-stdio-inheritance "app.app:create_app()"
