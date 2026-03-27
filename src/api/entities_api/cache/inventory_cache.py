import json
from typing import Dict, List, Optional

from redis.asyncio import Redis


class InventoryCache:
    """
    Manages the storage and retrieval of Network Device configurations.
    Scoped strictly by 'user_id' to allow inventory sharing across a user's Assistants.
    """

    def __init__(self, redis: Redis):
        self.redis = redis
        self.ttl = 86400  # 24 Hours

    # --- KEY GENERATION (User-Scoped) ---

    def _dev_key(self, user_id: str, hostname: str) -> str:
        """
        Key for the actual JSON data blob of a device.
        Format: net_eng:usr:{user_id}:inv:device:{hostname}
        """
        return f"net_eng:usr:{user_id}:inv:device:{hostname.lower()}"

    def _group_key(self, user_id: str, group: str) -> str:
        """
        Key for the Set containing hostnames belonging to a group.
        Format: net_eng:usr:{user_id}:inv:group:{group}
        """
        return f"net_eng:usr:{user_id}:inv:group:{group.lower()}"

    def _tenant_pattern(self, user_id: str) -> str:
        """
        Pattern to match ALL inventory keys for this specific User.
        """
        return f"net_eng:usr:{user_id}:inv:*"

    # --- PUBLIC METHODS ---

    async def clear_inventory(self, user_id: str) -> int:
        """
        Wipes all inventory data (devices and groups) for a specific user.
        Returns the number of Redis keys deleted.
        """
        pattern = self._tenant_pattern(user_id)
        keys_to_delete = []

        async for key in self.redis.scan_iter(match=pattern):
            keys_to_delete.append(key)

        if keys_to_delete:
            await self.redis.delete(*keys_to_delete)
            return len(keys_to_delete)

        return 0

    async def ingest_inventory(self, user_id: str, devices: List[Dict]) -> int:
        """
        Stores network devices into the specific User's namespace.
        """
        async with self.redis.pipeline() as pipe:
            for dev in devices:
                hostname = dev["host_name"]

                # 1. Store Device Data (Scoped to User)
                dev_key = self._dev_key(user_id, hostname)
                dev["owner_id"] = user_id

                await pipe.set(dev_key, json.dumps(dev), ex=self.ttl)

                # 2. Update Group Indexes (Scoped to User)
                if "groups" in dev and isinstance(dev["groups"], list):
                    for group in dev["groups"]:
                        g_key = self._group_key(user_id, group)
                        await pipe.sadd(g_key, hostname)
                        await pipe.expire(g_key, self.ttl)

                # 3. Always add to 'all' group for this scope
                all_key = self._group_key(user_id, "all")
                await pipe.sadd(all_key, hostname)
                await pipe.expire(all_key, self.ttl)

            await pipe.execute()

        return len(devices)

    async def search_by_group(self, user_id: str, group: str) -> List[Dict]:
        """
        Retrieves network devices for a specific group, strictly within the User's scope.
        """
        target_group_key = self._group_key(user_id, group)
        hostnames_bytes = await self.redis.smembers(target_group_key)

        if not hostnames_bytes:
            return []

        hostnames = [
            h.decode("utf-8") if isinstance(h, bytes) else h for h in hostnames_bytes
        ]

        async with self.redis.pipeline() as pipe:
            for host in hostnames:
                await pipe.get(self._dev_key(user_id, host))

            results = await pipe.execute()

        clean_results = []
        for raw_json in results:
            if raw_json:
                try:
                    clean_results.append(json.loads(raw_json))
                except json.JSONDecodeError:
                    continue

        return clean_results

    async def get_device(self, user_id: str, hostname: str) -> Optional[Dict]:
        """
        Retrieves a single device by hostname within the User's scope.
        """
        key = self._dev_key(user_id, hostname)
        raw_json = await self.redis.get(key)

        if raw_json:
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                return None
        return None
