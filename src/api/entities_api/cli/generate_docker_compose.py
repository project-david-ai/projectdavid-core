import os
from pathlib import Path


def generate_dev_docker_compose(output_path: str = "docker-compose.yml") -> None:
    """
    Generates the Project David local development docker-compose file.

    Includes optional training stack behind compose profiles.
    """

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
      test: ["CMD","mysqladmin","ping","-h","localhost"]
      interval: 10s
      timeout: 5s
      retries: 5
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
      - db
      - redis
      - qdrant
    networks:
      - my_custom_network

  training-api:
    image: thanosprime/projectdavid-core-training-api:latest
    build:
      context: .
      dockerfile: docker/training/Dockerfile.api
    container_name: training_api
    restart: unless-stopped
    profiles:
      - training
    env_file:
      - .env
    ports:
      - "9001:9001"
    networks:
      - my_custom_network

  training-worker:
    image: thanosprime/projectdavid-core-training-worker:latest
    build:
      context: .
      dockerfile: docker/training/Dockerfile
    container_name: training_worker
    restart: unless-stopped
    runtime: nvidia
    profiles:
      - training
    command: ["python","/app/worker.py"]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    networks:
      - my_custom_network

volumes:
  mysql_data:
  redis_data:
  qdrant_storage:

networks:
  my_custom_network:
    driver: bridge
"""

    Path(output_path).write_text(compose_yaml)
    print(f"✅ Dev docker compose generated → {output_path}")


if __name__ == "__main__":
    generate_dev_docker_compose()
