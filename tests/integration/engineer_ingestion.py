import json
import os
import sys

from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0. SDK Init & Environment Setup
# ------------------------------------------------------------------
load_dotenv()

client = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:9000/v1"),
    api_key=os.getenv("ENTITIES_API_KEY"),
)

# ------------------------------------------------------------------
# 1. Define the Network Topology (The "Mental Map")
# ------------------------------------------------------------------
network_inventory = [
    {
        "host_name": "core-router-01",
        "ip_address": "10.0.0.1",
        "platform": "cisco_ios",
        "groups": ["core", "hq", "critical"],
        "site": "New York HQ",
        "role": "Edge Router",
    },
    {
        "host_name": "dist-switch-01",
        "ip_address": "10.0.0.2",
        "platform": "cisco_nxos",
        "groups": ["distribution", "hq"],
        "site": "New York HQ",
        "role": "Aggregator",
    },
    {
        "host_name": "fw-primary",
        "ip_address": "10.0.0.254",
        "platform": "paloalto_panos",
        "groups": ["security", "firewall"],
        "site": "New York HQ",
        "role": "Perimeter Firewall",
    },
]


# ------------------------------------------------------------------
# 2. Ingest Inventory
# ------------------------------------------------------------------
def ingest():
    print(f"🚀 Uploading {len(network_inventory)} devices to The Engineer...")
    try:
        response = client.engineer.ingest_inventory(
            devices=network_inventory,
            clear_existing=True,
        )
        print("\n✅ Inventory uploaded successfully.")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"\n❌ Failed to ingest inventory: {e}")


# ------------------------------------------------------------------
# 3. Manual Device Lookup
# ------------------------------------------------------------------
def lookup_device(hostname: str):
    print(f"\n🔍 Looking up device: '{hostname}'...")
    try:
        result = client.engineer.get_device_info(hostname=hostname)
        if result:
            print("✅ Device found:")
            print(json.dumps(result, indent=2))
        else:
            print(f"⚠️  No device found with hostname '{hostname}'.")
    except Exception as e:
        print(f"❌ Lookup failed: {e}")


# ------------------------------------------------------------------
# 4. Manual Group Search
# ------------------------------------------------------------------
def lookup_group(group: str):
    print(f"\n🔍 Searching inventory for group: '{group}'...")
    try:
        results = client.engineer.search_inventory_by_group(group=group)
        if results:
            print(f"✅ Found {len(results)} device(s) in group '{group}':")
            print(json.dumps(results, indent=2))
        else:
            print(f"⚠️  No devices found in group '{group}'.")
    except Exception as e:
        print(f"❌ Group search failed: {e}")


# ------------------------------------------------------------------
# 5. Entrypoint — CLI dispatch
# ------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "ingest":
        ingest()
        lookup_device("core-router-01")
        lookup_group("core")

    elif args[0] == "device" and len(args) == 2:
        lookup_device(args[1])

    elif args[0] == "group" and len(args) == 2:
        lookup_group(args[1])

    else:
        print("Usage:")
        print(
            "  python script.py ingest                  # Upload inventory + run test searches"
        )
        print("  python script.py device <hostname>       # Look up a specific device")
        print("  python script.py group  <group_name>     # Search by group")
