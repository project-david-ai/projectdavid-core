from pathlib import Path


def generate_dev_docker_compose() -> None:

    # Use the current working directory (where the user runs the command)
    # This ensures it drops into the project root regardless of where the package is installed.
    output_path = Path.cwd() / "docker-compose.yml"

    if output_path.exists():
        print(f"⚠️  {output_path.name} already exists – generation skipped.")
        return

    compose_yaml = """\
services:

  db:
    image: mysql:8.0
    container_name: my_mysql_cosmic_catalyst
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: entities_db
      MYSQL_USER: api_user
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
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
      - SEARXNG_SECRET_KEY=${SEARXNG_SECRET_KEY}
    depends_on:
      - redis
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

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    profiles: ["ai"]
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
    profiles: [ "ai" ]
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN}
      - PYTORCH_ALLOC_CONF=expandable_segments:True
    volumes:
      - ${HF_CACHE_PATH}:/root/.cache/huggingface
      - ${SHARED_PATH}:/mnt/training_data
    ports:
      - "8001:8000"
    command: >
      --model ${VLLM_MODEL}
      ${VLLM_EXTRA_FLAGS}
      --dtype float16
      --max-model-len 2048
      --gpu-memory-utilization 0.5
    networks:
      - my_custom_network

  # ---------------------------------------------------------------------------
  # training-api — Fine-tuning REST API (no GPU required)
  # Opt-in: docker compose --profile training up training-api
  # ---------------------------------------------------------------------------
  training-api:
    image: thanosprime/projectdavid-core-training-api:latest
    build:
      context: .
      dockerfile: docker/training/Dockerfile
    container_name: training_api
    restart: unless-stopped
    profiles: ["training"]
    env_file:
      - .env
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
      - DEFAULT_SECRET_KEY=${DEFAULT_SECRET_KEY}
      - REDIS_URL=redis://redis:6379/0
      - ASSISTANTS_BASE_URL=http://api:9000
      - WORKER_API_KEY=${ADMIN_API_KEY}
      - SANDBOX_AUTH_SECRET=${SANDBOX_AUTH_SECRET}
      - SHARED_PATH=/mnt/training_data
      - PYTHONUNBUFFERED=1
      # training-api connects to Ray cluster via dashboard HTTP API only —
      # no direct GCS connection required. RAY_ADDRESS not needed here.
    ports:
      - "9001:9001"
    volumes:
      - ${SHARED_PATH:-./shared_data}:/mnt/training_data
      - ./src:/app/src
    depends_on:
      - redis
      - training-worker
    networks:
      - my_custom_network

  # ---------------------------------------------------------------------------
  # training-worker — GPU training runner (requires nvidia-container-toolkit)
  # Opt-in: docker compose --profile training up training-worker
  # ---------------------------------------------------------------------------
  training-worker:
    image: thanosprime/projectdavid-core-training-worker:latest
    build:
      context: .
      dockerfile: docker/training/Dockerfile
    container_name: training_worker
    restart: unless-stopped
    profiles: [ "training" ]
    env_file:
      - .env
    runtime: nvidia
    shm_size: '5gb'
    environment:
      - RAY_CLIENT_SERVER_PORT=10001
      - TRAINING_PROFILE=${TRAINING_PROFILE:-standard}
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://redis:6379/0
      - ASSISTANTS_BASE_URL=http://api:9000
      - WORKER_API_KEY=${ADMIN_API_KEY}
      - SHARED_PATH=/mnt/training_data
      - HF_TOKEN=${HF_TOKEN:-}
      - HF_HOME=/root/.cache/huggingface
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - PYTHONUNBUFFERED=1
      # Ray: unset = start as head node. Set to ray://<host>:10001 to join a cluster.
      - RAY_ADDRESS=${RAY_ADDRESS:-}
      - RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
    ports:
      - "8265:8265"    # Ray dashboard — http://localhost:8265
      - "10001:10001"  # Ray client protocol (external connections)
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ${SHARED_PATH:-./shared_data}:/mnt/training_data
      - ${HF_CACHE_PATH}:/root/.cache/huggingface
      - ./src:/app/src
    command: [ "python", "src/api/training/worker.py" ]
    depends_on:
      - redis
    networks:
      - my_custom_network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [ gpu ]

  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    container_name: fastapi_cosmic_catalyst
    restart: always
    env_file:
      - .env
    environment:
      - DATABASE_URL=${DATABASE_URL}
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
    ports:
      - "9000:9000"
    volumes:
      - ./src:/app/src
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
    networks:
      - my_custom_network

  sandbox:
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
      USER: "samba_user;${SMBCLIENT_PASSWORD}"
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
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
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

    output_path.write_text(compose_yaml)
    print("✅  docker-compose.yml generated (AI + training are opt-in profiles).")


if __name__ == "__main__":
    generate_dev_docker_compose()
