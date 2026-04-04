from datetime import datetime
from typing import Any, Optional, Union


def datetime_to_iso(value: Optional[datetime]) -> Optional[str]:
    """Convert a datetime object to an ISO-8601 string."""
    return value.isoformat() if value else None


def iso_to_datetime(value: Optional[str]) -> Optional[datetime]:
    """Convert an ISO-8601 string to a datetime object."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def convert_dict_keys_to_snake_case(data: dict) -> dict:
    """Convert dictionary keys from camelCase to snake_case."""
    return {_camel_to_snake(key): value for key, value in data.items()}


def _camel_to_snake(s: str) -> str:
    """Helper function to convert camelCase to snake_case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in s]).lstrip("_")


def convert_dict_keys_to_camel_case(data: dict) -> dict:
    """Convert dictionary keys from snake_case to camelCase."""
    return {_snake_to_camel(key): value for key, value in data.items()}


def _snake_to_camel(s: str) -> str:
    """Helper function to convert snake_case to camelCase."""
    parts = s.split("_")
    return parts[0] + "".join((part.title() for part in parts[1:]))


def convert_nested_dict(data: dict, key_converter: callable) -> dict:
    """Recursively convert keys in a nested dictionary."""
    return {
        key_converter(key): (
            convert_nested_dict(value, key_converter)
            if isinstance(value, dict)
            else value
        )
        for key, value in data.items()
    }
