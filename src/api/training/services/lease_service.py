# src/api/training/services/lease_service.py
"""
lease_service.py

Redis-backed singleton lease for the training-api.

Ensures only one training-api instance acts as Master at any time,
preventing concurrent metadata writes and job dispatch races.

Usage:
    from src.api.training.services.lease_service import acquire_api_lease, renew_api_lease

    r = get_redis_client()
    if not acquire_api_lease(r, instance_id):
        sys.exit(1)
"""


def acquire_api_lease(r, instance_id: str, timeout: int = 30) -> bool:
    """
    Attempt to claim the Master lease.

    Uses SET NX — only succeeds if no other instance currently holds the key.
    Returns True if the lease was acquired, False if another instance is Master.
    """
    return r.set("cluster:active_training_api", instance_id, ex=timeout, nx=True)


def renew_api_lease(r, instance_id: str, timeout: int = 30) -> bool:
    """
    Renew the Master lease atomically.

    Uses a Lua script to check-and-extend: only renews if this instance
    still owns the lease. Returns True if renewed, False if the lease was
    lost to another instance (triggers hard shutdown in the caller).
    """
    script = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("expire", KEYS[1], ARGV[2])
    else
        return 0
    end
    """
    return r.eval(script, 1, "cluster:active_training_api", instance_id, timeout)
