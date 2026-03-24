#!/usr/bin/env bash
# Starts the main command (Supervisor → Uvicorn)
# Schema sync is handled automatically at startup by ensure_schema()
set -e

echo "🚀 Starting Supervisor (which will run Uvicorn)…"
exec "$@"