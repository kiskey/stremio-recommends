# Dockerfile

# Use a modern, supported base image
FROM python:3.9-slim-bookworm

WORKDIR /usr/src/app

# Copy and install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data artifacts into the image
COPY artifacts ./artifacts

# Copy the application code
COPY main.py .

# --- NEW ENTRYPOINT LOGIC ---
# Copy the entrypoint script into the container
COPY entrypoint.sh .
# Make the script executable
RUN chmod +x ./entrypoint.sh

ENV ARTIFACTS_DIR=/usr/src/app/artifacts
EXPOSE 5000

# Set the entrypoint script to run on container start
ENTRYPOINT ["./entrypoint.sh"]

# The CMD is now passed as an argument to the entrypoint script
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:create_app()"]
