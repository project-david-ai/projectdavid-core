docker compose exec api alembic revision --autogenerate -m "add engineer column to assistants"
docker compose exec api alembic revision --autogenerate -m "add BatfishSnapshot to models"
docker compose exec api alembic revision --autogenerate -m "add ID to BatfishSnapshot to models"
docker compose exec api alembic revision --autogenerate -m "add owner_id to Thread table"
docker compose exec api alembic revision --autogenerate -m "Remove assistant ---> vector_store relationship"
docker compose exec api alembic revision --autogenerate -m "Remove thread ---> vector_store relationship"
docker compose exec api alembic revision --autogenerate -m "Add soft delete to Files"
platform-api docker-manager --mode both --exclude ollama --exclude vllm
docker compose exec api alembic revision --autogenerate -m "Add soft delete to VectorStore"
docker compose exec api alembic revision --autogenerate -m "Add fine tuning tables"
docker compose exec api alembic revision --autogenerate -m "Move fine tuning tables to training root instance of models.py"
docker compose exec api alembic revision --autogenerate -m "Implement Fine tuning tables in training.models"




docker compose exec api alembic revision --autogenerate -m "Remove training data from models.py"

docker compose exec api alembic revision --autogenerate -m "Add training data to training.models2"

docker compose exec api alembic revision --autogenerate -m "Add fine tuning tables"




docker compose exec api alembic revision --autogenerate -m "updated_at column to match other models"

docker compose exec api alembic revision --autogenerate -m "adding updated_at and deleted_at to the TrainingJobRead"



docker compose exec api alembic revision --autogenerate -m "Add cluster management tables and fields"
