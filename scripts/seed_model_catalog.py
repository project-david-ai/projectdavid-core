from src.api.training.db.database import SessionLocal
from src.api.training.models.models import BaseModel

MODELS = [
    {
        "id": "unsloth/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 (1B)",
        "family": "llama",
        "parameter_count": "1B",
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen 2.5 (1.5B)",
        "family": "qwen",
        "parameter_count": "1.5B",
    },
]


def seed():
    db = SessionLocal()
    for m in MODELS:
        exists = db.query(BaseModel).filter(BaseModel.id == m["id"]).first()
        if not exists:
            db.add(BaseModel(**m))
    db.commit()
    db.close()
    print("✅ Model Catalog Seeded.")


if __name__ == "__main__":
    seed()
