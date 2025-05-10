# Use CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash developer
WORKDIR /home/developer/app
ENV PYTHONPATH="/home/developer/app:${PYTHONPATH}"
RUN chown -R developer:developer /home/developer/app

# Switch to non-root user
USER developer

# Install Python dependencies
COPY --chown=developer:developer requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy project files
COPY --chown=developer:developer . .

# Set default command
CMD ["/bin/bash"] 