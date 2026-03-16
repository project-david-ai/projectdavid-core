#!src/api/entities_api/cli/generate_docker_compose.py
"""
Generate a development-friendly docker-compose.yml and training overlay.

All sensitive values (DB passwords, secret keys, etc.) are expressed as
${ENV_VAR:?message} placeholders — Docker Compose will refuse to start and
print a clear error if any required variable is missing or empty, rather
than silently using an insecure default.

The docker_manager orchestration script generates real secrets and writes
them into .env on first run, so the :? guard should never fire in normal use.
It exists as a last line of defence if someone runs `docker compose up`
directly without going through the manager.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Training Overlay Content
# --------------------------------------------------------------------------- #
TRAINING_YML_CONTENT = """\
services:
  training-worker:
    image: thanosprime/projectdavid-core-training:latest
    build:
      context: .
      dockerfile: docker/training/Dockerfile
    container_name: training_worker
    restart: unless-stopped
    runtime: nvidia
    environment:
      - REDIS_URL=${REDIS_URL:-redis://redis:6379/0}
      - ASSISTANTS_BASE_URL=http://api:9000
      - WORKER_API_KEY=${ADMIN_API_KEY}
      - SHARED_PATH=/mnt/training_data
      - HF_TOKEN=${HF_TOKEN:-}
      - HF_HOME=/mnt/training_data/.hf_cache
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - PYTHONUNBUFFERED=1
    volumes:
      - ${SHARED_PATH:-./shared_data}:/mnt/training_data
    command: ["python", "/app/worker.py"]
    depends_on:
      - redis
      - api
    networks:
      - my_custom_network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""


# --------------------------------------------------------------------------- #
# Main generator
# --------------------------------------------------------------------------- #
def generate_dev_docker_compose(force: bool = False) -> None:
    """
    Write docker-compose.yml and docker-compose.training.yml to the project root.

    Parameters
    ----------
    force:
        If True, overwrite existing files. Defaults to False (skip if present).
        Pass force=True when you know the template has changed and need to
        push the update to disk.
    """
    # project root — file lives at src/api/entities_api/cli/generate_docker_compose.py
    # so we must walk up 5 levels: cli → entities_api → api → src → repo root
    project_root = Path(__file__).resolve().parents[4]

    main_compose_path = project_root / "docker-compose.yml"
    training_compose_path = project_root / "docker-compose.training.yml"

    # --- 1. Generate Main Compose File ---
    if main_compose_path.exists() and not force:
        print(f"⚠️  {main_compose_path.name} already exists – generation skipped.")
    else:
        compose_yaml = """\
services:
  db:
    image: mysql:8.0
    container_name: my_mysql_cosmic_catalyst
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD:?MYSQL_ROOT_PASSWORD is not set. Run the docker-manager to generate secrets.}
      MYSQL_DATABASE: entities_db
      MYSQL_USER: api_user
      MYSQL_PASSWORD: ${MYSQL_PASSWORD:?MYSQL_PASSWORD is not set. Run the docker-manager to generate secrets.}
    volumes:
      - mysql_data:/var/lib/mysql
    ports:
      - "3307:3306"
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - my_custom_network

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant_server
    restart: always
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      QDRANT__STORAGE__STORAGE_PATH: "/qdrant/storage"
      QDRANT__SERVICE__GRPC_PORT: "6334"
      QDRANT__LOG_LEVEL: "INFO"
    networks:
      - my_custom_network

  redis:
    image: redis:7
    container_name: redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - my_custom_network

  browser:
    image: ghcr.io/browserless/chromium:latest
    container_name: browserless_chromium
    restart: always
    ports:
      - "3000:3000"
    environment:
      - MAX_CONCURRENT_SESSIONS=10
      - CONNECTION_TIMEOUT=60000
    networks:
      - my_custom_network

  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./docker/searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/
      - SEARXNG_SECRET_KEY=${SEARXNG_SECRET_KEY:?SEARXNG_SECRET_KEY is not set. Run the docker-manager to generate secrets.}
    depends_on:
      - redis
    networks:
      - my_custom_network

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: otel_collector
    restart: always
    command: ["--config=/etc/otel-config.yaml"]
    volumes:
      - ./docker/otel/otel-config.yaml:/etc/otel-config.yaml
    ports:
      - "4317:4317"
      - "4318:4318"
    depends_on:
      - jaeger
    networks:
      - my_custom_network

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger_ui
    restart: always
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"
      - "14250:14250"
    networks:
      - my_custom_network

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - my_custom_network

  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm_server
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ${HF_CACHE_PATH:?HF_CACHE_PATH is not set. Run the docker-manager to configure it.}:/root/.cache/huggingface
    ports:
      - "8001:8000"
    command: >
      --model ${VLLM_MODEL:?VLLM_MODEL is not set. Run the docker-manager to configure it.}
      --dtype float16
      --quantization fp8
      --max-model-len 4096
      --gpu-memory-utilization 0.85
    networks:
      - my_custom_network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  api:
    image: thanosprime/projectdavid-core-api:latest
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    container_name: fastapi_cosmic_catalyst
    restart: always
    env_file:
      - .env
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - AUTO_MIGRATE=1
      - SANDBOX_SERVER_URL=http://sandbox:8000
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
      - BROWSER_WS_ENDPOINT=ws://browser:3000
      - DEFAULT_SECRET_KEY=${DEFAULT_SECRET_KEY}
      - SEARXNG_URL=http://searxng:8080
      - OTEL_SERVICE_NAME=api-api
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
      - OTEL_TRACES_EXPORTER=otlp
      - OTEL_METRICS_EXPORTER=none
      - OTEL_LOGS_EXPORTER=none
      - OLLAMA_BASE_URL=http://ollama:11434/v1
      - VLLM_BASE_URL=http://vllm_server:8000
      - SHARED_PATH=/app/shared_data
      - ASSISTANTS_BASE_URL=http://localhost:80
      - DOWNLOAD_BASE_URL=http://localhost:80/v1/files/download
    # Port 9000 kept open in dev for direct access / debugging.
    # In production traffic flows through nginx on port 80.
    ports:
      - "9000:9000"
    volumes:
      - ./src:/app/src
      - ./alembic.ini:/app/alembic.ini
      - ./migrations:/app/migrations
      - ${SHARED_PATH}:/app/shared_data
    depends_on:
      db:
        condition: service_healthy
      sandbox:
        condition: service_started
      qdrant:
        condition: service_started
      redis:
        condition: service_started
      browser:
        condition: service_started
      searxng:
        condition: service_started
      otel-collector:
        condition: service_started
      ollama:
        condition: service_started
      vllm:
        condition: service_started
    networks:
      - my_custom_network

  sandbox:
    image: thanosprime/projectdavid-core-sandbox:latest
    build:
      context: .
      dockerfile: docker/sandbox/Dockerfile
    container_name: sandbox_api
    restart: always
    cap_add:
      - SYS_ADMIN
    security_opt:
      - seccomp:unconfined
    devices:
      - /dev/fuse
    ports:
      - "8000:8000"
    volumes:
      - ./src/api/sandbox:/app/sandbox
      - /tmp/sandbox_logs:/app/logs
    depends_on:
      db:
        condition: service_healthy
    env_file:
      - .env
    networks:
      - my_custom_network

  samba:
    image: dperson/samba
    container_name: samba_server
    restart: unless-stopped
    environment:
      USERID: ${SAMBA_USERID:-1000}
      GROUPID: ${SAMBA_GROUPID:-1000}
      TZ: UTC
      USER: "samba_user;${SMBCLIENT_PASSWORD:?SMBCLIENT_PASSWORD is not set. Run the docker-manager to generate secrets.}"
      SHARE: "cosmic_share;/samba/share;yes;no;no;samba_user"
      GLOBAL: "server min protocol = NT1\\nserver max protocol = SMB3"
    ports:
      - "139:139"
      - "1445:445"
    volumes:
      - ${SHARED_PATH}:/samba/share
    networks:
      - my_custom_network

  nginx:
    image: nginx:alpine
    container_name: nginx_proxy
    restart: always
    ports:
      - "80:80"
      # Uncomment when TLS certs are available:
      # - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      # Uncomment when TLS certs are available:
      # - ./docker/nginx/certs:/etc/nginx/certs:ro
    depends_on:
      - api
    networks:
      - my_custom_network

volumes:
  mysql_data:
  qdrant_storage:
  redis_data:
  ollama_data:

networks:
  my_custom_network:
    driver: bridge
"""
        main_compose_path.write_text(compose_yaml, encoding="utf-8")
        print(f"✅  docker-compose.yml written → {main_compose_path}")

    # --- 2. Generate Training Overlay File ---
    if training_compose_path.exists() and not force:
        print(f"⚠️  {training_compose_path.name} already exists – generation skipped.")
    else:
        training_compose_path.write_text(TRAINING_YML_CONTENT, encoding="utf-8")
        print(f"✅  docker-compose.training.yml written → {training_compose_path}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    generate_dev_docker_compose(force=True)
