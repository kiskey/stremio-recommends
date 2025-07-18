# Dockerfile

# Use a modern, supported base image
FROM python:3.9-slim-bookworm

WORKDIR /usr/src/app

# Copy and install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts and data into the image
COPY artifacts ./artifacts
COPY main.py .
COPY trakt_sync.py .
COPY entrypoint.sh .

# Make the entrypoint script executable
RUN chmod +x ./entrypoint.sh

# Set environment variables that both scripts use
ENV ARTIFACTS_DIR=/usr/src/app/artifacts
ENV HISTORY_DB_PATH=/usr/src/app/persistent_data/watch_history.db

EXPOSE 5000

# Set our new script as the entrypoint for the container
ENTRYPOINT ["./entrypoint.sh"]
