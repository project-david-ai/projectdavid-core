from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.entities_api.routers import api_router


def test_v1_health_is_mounted_and_reports_database_ready():
    app = FastAPI()
    app.include_router(api_router, prefix="/v1")

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "database": True,
        "status": "healthy",
    }
