#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- PRE-FLIGHT CHECK ---
# Verify that all critical artifact files exist before starting the server.
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
echo "✅ All data artifacts found. Starting Gunicorn server..."

# Execute the command passed to this script (which will be the gunicorn command from the Dockerfile's CMD).
# 'exec' replaces the shell process with the gunicorn process, which is more efficient.
exec "$@"
