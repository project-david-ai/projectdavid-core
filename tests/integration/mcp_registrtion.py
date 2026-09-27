import os

from dotenv import load_dotenv
from projectdavid import Entity


load_dotenv(".tests.env")


base_url = os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL")
api_key = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")
assistant_id = os.getenv("ASSISTANT_ID")

assert base_url, "PROJECT_DAVID_PLATFORM_BASE_URL is not set"
assert api_key, "DEV_PROJECT_DAVID_CORE_TEST_USER_KEY is not set"
assert assistant_id, "ASSISTANT_ID is not set"


client = Entity(
    base_url=base_url,
    api_key=api_key,
)


# ---------------------------------------------------------
# 1. Register a real public MCP server
# ---------------------------------------------------------

server = client.mcp.create_server(
    name="deepwiki",
    url="https://mcp.deepwiki.com/mcp",
)

print("\nMCP server:")
print(server)


# ---------------------------------------------------------
# 2. Discover its actual advertised tools
# ---------------------------------------------------------

tools = client.mcp.discover_tools(server)

print("\nAvailable MCP tools:")

for tool in tools:
    print(
        f"- name={tool.name!r}"
        f"  canonical_id={tool.canonical_id!r}"
        f"  provider_name={tool.provider_name!r}"
        f"  description={tool.description!r}"
    )


# ---------------------------------------------------------
# 3. Select a deliberate subset
# ---------------------------------------------------------

selected = tools.select(
    "read_wiki_structure",
    "read_wiki_contents",
)

print("\nSelected MCP tools:")

for tool in selected:
    print(f"- {tool.name} -> {tool.provider_name}")


# ---------------------------------------------------------
# 4. Attach them to an existing assistant
# ---------------------------------------------------------

attached = client.mcp.attach_tools(
    assistant_id=assistant_id,
    tools=selected,
)

print("\nAttached:")

for tool in attached:
    print(
        f"- remote_name={tool.remote_name!r}" f"  provider_name={tool.provider_name!r}"
    )


# ---------------------------------------------------------
# 5. Retrieve assistant and prove tool_configs changed
# ---------------------------------------------------------

assistant = client.assistants.retrieve_assistant(
    assistant_id,
)

print("\nAssistant:")
print(assistant)

print("\nAssistant tools:")

for tool in assistant.tools:
    print(tool)
