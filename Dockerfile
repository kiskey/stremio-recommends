# Dockerfile

# Use a modern, supported base image
FROM python:3.9-slim-bookworm

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV ARTIFACTS_DIR=/usr/src/app/artifacts

EXPOSE 5000

# --- UPDATED CMD INSTRUCTION ---
# This tells Gunicorn to look inside the 'main' module for a function
# named 'create_app' and to call it to get the Flask app object.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:create_app()"]
