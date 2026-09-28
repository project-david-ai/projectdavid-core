import os

from dotenv import load_dotenv
from projectdavid import Entity


load_dotenv(".tests.env")


base_url = os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL")
api_key = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")
assistant_id = os.getenv("ASSISTANT_ID")
github_token = os.getenv("GITHUB_MCP_PAT")

assert base_url, "PROJECT_DAVID_PLATFORM_BASE_URL is not set"
assert api_key, "DEV_PROJECT_DAVID_CORE_TEST_USER_KEY is not set"
assert assistant_id, "ASSISTANT_ID is not set"
assert github_token, "GITHUB_MCP_PAT is not set"


client = Entity(
    base_url=base_url,
    api_key=api_key,
)

# ---------------------------------------------------------
# 1. List MCP servers per user
# ---------------------------------------------------------
list_mcp_servers = client.mcp.list_servers()
print(f"list_servers={list_mcp_servers}")

# ---------------------------------------------------------
# 2. Retrieve MCP server
# ---------------------------------------------------------
retrieve_server = client.mcp.retrieve_server(server_id="mcpreg_ldCtbN356EvBJ0EZfFtExc")
print(f"retrieve_server={retrieve_server}")

# ---------------------------------------------------------
# 3. List MCP sourced tools.
# ---------------------------------------------------------
assistant_tools = client.mcp.list_assistant_tools(assistant_id=assistant_id)
print(f"assistant_tools={assistant_tools}")
