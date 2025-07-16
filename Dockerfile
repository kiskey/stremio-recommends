# Dockerfile

# Use a modern, supported base image
FROM python:3.9-slim-bookworm

WORKDIR /usr/src/app

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW STEP: Copy the data artifacts into the image ---
# This assumes the 'artifacts' directory exists in the build context
COPY artifacts ./artifacts

# Copy the application code
COPY main.py .

# Set environment variable for the artifacts directory inside the container
ENV ARTIFACTS_DIR=/usr/src/app/artifacts

EXPOSE 5000

# The CMD remains the same
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:create_app()"]
