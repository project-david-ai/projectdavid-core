# scripts/seed_model_catalog.py
from projectdavid_orm.ormInterface import BaseModel

from src.api.entities_api.db.database import SessionLocal
from src.api.training.constants.models import SUPPORTED_BASE_MODELS


def seed():
    db = SessionLocal()
    for m in SUPPORTED_BASE_MODELS:
        exists = db.query(BaseModel).filter(BaseModel.id == m["id"]).first()
        if not exists:
            db.add(BaseModel(**m))
    db.commit()
    db.close()
    print("✅ Model Catalog Seeded.")


if __name__ == "__main__":
    seed()
