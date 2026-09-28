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
# 1. Register a real authenticated remote MCP server
# ---------------------------------------------------------

server = client.mcp.create_server(
    name="github-auth-test",
    url="https://api.githubcopilot.com/mcp/",
    bearer_token=github_token,
)

print("\nMCP server:")
print(server)

assert server.auth_type == "bearer"

server_dump = server.model_dump()

assert "token" not in server_dump
assert "credential_id" not in server_dump
assert "encrypted_payload" not in server_dump

print("\nAuthenticated registration:")
print(f"- id={server.id!r}")
print(f"- name={server.name!r}")
print(f"- auth_type={server.auth_type!r}")
print("- credential_exposed=False")


# ---------------------------------------------------------
# 2. Discover tools WITHOUT supplying the token again
#
# This is the important Auth-1 proof:
#
# registration
#     -> credential persisted/encrypted by Core
#     -> discovery resolves registration credential
#     -> Core sends Authorization: Bearer <token>
# ---------------------------------------------------------

tools = client.mcp.discover_tools(server)

assert len(tools) > 0, "GitHub MCP returned no tools"

print(f"\nAvailable MCP tools: {len(tools)}")

for tool in tools:
    print(
        f"- name={tool.name!r}"
        f"  canonical_id={tool.canonical_id!r}"
        f"  provider_name={tool.provider_name!r}"
        f"  description={tool.description!r}"
    )


# ---------------------------------------------------------
# 3. Select a deliberate, read-oriented subset
# ---------------------------------------------------------

selected = tools.select(
    "get_file_contents",
    "search_repositories",
)

print("\nSelected MCP tools:")

for tool in selected:
    print(f"- {tool.name!r}" f" -> {tool.provider_name!r}")


# ---------------------------------------------------------
# 4. Attach authenticated MCP tools to existing assistant
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


# ---------------------------------------------------------
# 6. Final assertions
# ---------------------------------------------------------

attached_names = {tool.remote_name for tool in attached}

assert "get_file_contents" in attached_names
assert "search_repositories" in attached_names

print("\n========================================")
print("AUTHENTICATED MCP INTEGRATION = PASS")
print("========================================")
print(f"server_id={server.id}")
print(f"auth_type={server.auth_type}")
print(f"discovered_tools={len(tools)}")
print("token_resupplied_for_discovery=NO")
print("token_resupplied_for_attachment=NO")
print("credential_exposed_by_registration=NO")
print("attached=get_file_contents,search_repositories")
