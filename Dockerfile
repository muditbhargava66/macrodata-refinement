# Use Python 3.10 as the base image
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Expose the API port
EXPOSE 8000

# Create a non-root user to run the application
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

# Use an entrypoint script
COPY ./scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "-m", "mdr.api.cli", "--help"]