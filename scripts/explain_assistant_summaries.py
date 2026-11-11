"""Read-only plans on the configured Core DB; run after migration."""

import argparse
import json

from sqlalchemy import inspect

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.services.assistant_summaries import (
    SummaryFilters,
    summary_query,
)


def describe(node):
    if isinstance(node, dict):
        if "table_name" in node:
            print(
                {
                    k: node[k]
                    for k in (
                        "table_name",
                        "access_type",
                        "possible_keys",
                        "key",
                        "rows_examined_per_scan",
                    )
                    if k in node
                }
            )
        if node.get("using_filesort"):
            print("filesort")
        for value in node.values():
            describe(value)
    elif isinstance(node, list):
        for value in node:
            describe(value)


p = argparse.ArgumentParser()
p.add_argument("--user-id", required=True)
p.add_argument("--model", default="gpt-4")
p.add_argument("--prefix", default="Portal")
args = p.parse_args()
with SessionLocal() as db:
    conn = db.connection()
    schema = inspect(conn)
    print("Assistant indexes:", [i["name"] for i in schema.get_indexes("assistants")])
    print(
        "Association PK:",
        schema.get_pk_constraint("user_assistants")["constrained_columns"],
    )
    for label, filters in [
        ("default", SummaryFilters()),
        ("model", SummaryFilters(model=args.model)),
        ("mode", SummaryFilters(mode="standard")),
        ("prefix", SummaryFilters(name_prefix=args.prefix)),
    ]:
        print(label)
        sql = summary_query(args.user_id, filters).compile(
            conn, compile_kwargs={"literal_binds": True}
        )
        describe(
            json.loads(
                conn.exec_driver_sql("EXPLAIN FORMAT=JSON " + str(sql)).scalar_one()
            )
        )
