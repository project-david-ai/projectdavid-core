import json
import os
import sys
from unittest.mock import MagicMock

# -------------------------------------------------------------------------
# 1. SETUP DUMMY ENVIRONMENT
#    We set these before importing the app to prevent config validation errors.
# -------------------------------------------------------------------------
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["ENTITIES_API_KEY"] = "dummy_key"
os.environ["SIGNED_URL_SECRET"] = "dummy_secret"
os.environ["DOWNLOAD_BASE_URL"] = "http://localhost:9000"
os.environ["HYPERBOLIC_API_KEY"] = "dummy"
os.environ["TOGETHER_API_KEY"] = "dummy"

# -------------------------------------------------------------------------
# 2. MOCK BLOCKING MODULES
#    Your main.py calls `wait_for_databases()` on import.
#    We must mock the database module BEFORE importing main to skip that check.
# -------------------------------------------------------------------------
mock_db_module = MagicMock()
mock_db_module.wait_for_databases = lambda: print(
    ">> [MOCK] Skipping DB Connection Check"
)
mock_db_module.engine = MagicMock()
mock_db_module.SessionLocal = MagicMock()

# Inject the mock into sys.modules so Python uses it instead of the real file
sys.modules["src.api.entities_api.db.database"] = mock_db_module

# -------------------------------------------------------------------------
# 3. IMPORT APP AND GENERATE SPEC
# -------------------------------------------------------------------------
try:
    from fastapi.openapi.utils import get_openapi

    from src.api.entities_api.app import create_app

    print(">> Initializing FastAPI App (Schema Mode)...")
    app = create_app(init_db=False)  # We pass False so it doesn't try to create tables

    print(">> Generating OpenAPI JSON...")

    # Generate the raw schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    # ---------------------------------------------------------------------
    # 4. STAINLESS OPTIMIZATION (OPTIONAL BUT RECOMMENDED)
    #    Stainless uses 'operationId' to name SDK methods.
    #    FastAPI defaults to function names (e.g., 'create_run').
    #    This logic ensures method names are clean.
    # ---------------------------------------------------------------------
    for path, methods in openapi_schema["paths"].items():
        for method, details in methods.items():
            # If function is named 'create_run', operationId becomes 'create_run'
            # This results in SDK usage: client.runs.create_run()
            # You can manually rename these here if you prefer client.runs.create()
            if "operationId" not in details:
                # Fallback if FastAPI didn't set it (rare)
                details["operationId"] = (
                    details.get("summary", "").replace(" ", "_").lower()
                )

    # ---------------------------------------------------------------------
    # 5. WRITE TO FILE
    # ---------------------------------------------------------------------
    output_filename = "../openapi.json"
    with open(output_filename, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"\n✅ SUCCESS! Schema saved to: {os.path.abspath(output_filename)}")
    print("   Upload this file to the Stainless dashboard.")

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print(
        "   Make sure you are running this from the project root and your virtualenv is active."
    )
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
