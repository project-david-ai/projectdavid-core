from pathlib import Path


def generate_dev_docker_compose() -> None:

    project_root = Path(__file__).resolve().parents[4]
    output_path = project_root / "docker-compose.yml"

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
      test: ["CMD","mysqladmin","ping","-h","localhost"]
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
    profiles: ["ai"]
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ${HF_CACHE_PATH}:/root/.cache/huggingface
    ports:
      - "8001:8000"
    command: >
      --model ${VLLM_MODEL}
      --dtype float16
      --max-model-len 4096
    networks:
      - my_custom_network


  training-api:
    image: thanosprime/projectdavid-core-training-api:latest
    container_name: training_api
    restart: unless-stopped
    profiles: ["training"]
    env_file:
      - .env
    ports:
      - "9001:9001"
    depends_on:
      - redis
    networks:
      - my_custom_network


  training-worker:
    image: thanosprime/projectdavid-core-training-worker:latest
    container_name: training_worker
    restart: unless-stopped
    profiles: ["training"]
    runtime: nvidia
    depends_on:
      - redis
    networks:
      - my_custom_network


  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    container_name: fastapi_cosmic_catalyst
    restart: always
    env_file:
      - .env
    ports:
      - "9000:9000"
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
    ports:
      - "8000:8000"
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
    print("✅ FULL docker-compose generated (AI + training are opt-in profiles).")


if __name__ == "__main__":
    generate_dev_docker_compose()
