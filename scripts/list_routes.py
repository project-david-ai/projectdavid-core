import os

from dotenv import load_dotenv
from fastapi.routing import APIRoute
from tabulate import tabulate

from src.api.entities_api.app import create_app

load_dotenv()
os.environ["DISABLE_DB_INIT"] = "1"
app = create_app(init_db=False)
routes_data = []
for route in app.routes:
    if isinstance(route, APIRoute):
        method_list = ", ".join(route.methods - {"HEAD", "OPTIONS"})
        routes_data.append(
            [route.path, method_list, route.name or "", route.summary or ""]
        )
headers = ["Path", "Method(s)", "Name", "Summary"]
markdown_table = tabulate(routes_data, headers, tablefmt="github")
print("# 📌 FastAPI Routes (Markdown View)\n")
print(markdown_table)
