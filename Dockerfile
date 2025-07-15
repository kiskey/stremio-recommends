# Use a lean official Python image
FROM python:3.9-slim-buster

# Set working directory in the container
WORKDIR /usr/src/app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py .

# Set environment variable for the artifacts directory inside the container
ENV ARTIFACTS_DIR=/usr/src/app/artifacts

# Expose the port the app runs on
EXPOSE 5000

# Run the app using a production-grade WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
