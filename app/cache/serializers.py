# app/cache/serializers.py
"""
Serialization utilities for cache values.
Handles JSON encoding/decoding with special float value handling.
"""

from typing import Any, Optional
import logging

import orjson

logger = logging.getLogger(__name__)


def clean_float_values(obj: Any) -> Any:
    """
    Recursively clean NaN and inf values from nested structures.

    Replaces NaN, inf, -inf with None to ensure JSON compatibility.

    Args:
        obj: Object to clean (dict, list, or primitive)

    Returns:
        Cleaned object with NaN/inf replaced by None
    """
    if isinstance(obj, dict):
        return {k: clean_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_float_values(item) for item in obj]
    elif isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    else:
        return obj


def serialize(obj: Any) -> Optional[str]:
    """
    Serialize Python object to JSON string.

    Automatically cleans NaN/inf values before serialization.

    Args:
        obj: Object to serialize

    Returns:
        JSON string or None on error
    """
    try:
        cleaned = clean_float_values(obj)
        return orjson.dumps(cleaned)
    except Exception as e:
        logger.warning(f"Serialization failed: {e}")
        return None


def deserialize(json_str: str) -> Optional[Any]:
    """
    Deserialize JSON string to Python object.

    Args:
        json_str: JSON string to deserialize

    Returns:
        Deserialized object or None on error
    """
    try:
        return orjson.loads(json_str)
    except Exception as e:
        logger.warning(f"Deserialization failed: {e}")
        return None
