#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Stremio Recommends Entrypoint ---"

# --- PRE-FLIGHT CHECK (from your existing script) ---
# Verify that all critical artifact files exist before starting anything.
# We use the ARTIFACTS_DIR environment variable set in the Dockerfile.

if [ ! -f "$ARTIFACTS_DIR/enriched_titles.pkl" ] || \
   [ ! -f "$ARTIFACTS_DIR/tfidf_vectorizer.pkl" ] || \
   [ ! -f "$ARTIFACTS_DIR/tfidf_matrix.pkl" ]; then
  
  # Print a loud, clear error message to standard error.
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
  echo "!!! FATAL ERROR: One or more critical data artifacts not found!" >&2
  echo "!!! Searched in directory: $ARTIFACTS_DIR" >&2
  echo "!!! The container will now exit." >&2
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
  
  # Exit with a non-zero status code to signal a fatal error.
  exit 1
fi

# If all checks pass, print a success message and proceed.
echo "✅ All data artifacts found."

# --- DUAL PROCESS STARTUP ---

# Start the Trakt Sync Worker in the background.
# The '&' symbol sends the process to the background.
echo "✅ Starting Trakt Sync Worker in the background..."
python trakt_sync.py &

# Start the Gunicorn web server in the foreground.
# This will be the main process that keeps the container running.
# 'exec' replaces the shell process with the gunicorn process, which is a best practice.
echo "✅ Starting Gunicorn web server in the foreground..."
exec gunicorn --bind 0.0.0.0:5000 "main:create_app()"
