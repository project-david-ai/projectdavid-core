#!/usr/bin/env bash
set -e

echo "🔄 Running database migrations..."
cd /app && alembic upgrade head

echo "🚀 Starting Supervisor (which will run Uvicorn)..."
exec "$@"
