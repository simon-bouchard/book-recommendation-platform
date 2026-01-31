# app/cache

Generic Redis caching infrastructure for the application.

## Overview

This module provides a reusable caching layer that can be used across different parts of the application. It includes connection management, automatic serialization, and graceful degradation if Redis is unavailable.

## Components

### `client.py`
Redis connection management with singleton pattern and connection pooling.

**Key features:**
- Connection pooling for performance
- Automatic reconnection handling
- Graceful degradation if Redis unavailable
- Thread-safe singleton access

### `serializers.py`
JSON serialization with special handling for problematic float values (NaN, inf).

**Key features:**
- Automatic cleaning of NaN/inf values
- Recursive handling of nested structures
- Error handling with logging

### `decorators.py`
Generic `@cached` decorator for easy function result caching.

**Key features:**
- Works with both sync and async functions
- Automatic serialization/deserialization
- Cache hit/miss logging
- Graceful fallback on errors

## Usage

### Basic caching

```python
from app.cache import cached

@cached(
    key_func=lambda user_id: f"user:profile:{user_id}",
    ttl=3600
)
def get_user_profile(user_id: int) -> dict:
    # Expensive computation
    return {"id": user_id, "name": "..."}
```

### Configuration

Set environment variables:
```bash
REDIS_HOST=localhost      # Default: localhost
REDIS_PORT=6379           # Default: 6379
REDIS_DB=0                # Default: 0
REDIS_PASSWORD=secret     # Default: None
```

Or configure programmatically:
```python
from app.cache import get_redis_client

client = get_redis_client(
    host="localhost",
    port=6379,
    db=1,
    password=None
)
```

### Direct client usage

```python
from app.cache import get_redis_client

client = get_redis_client()

# Set value with TTL
client.set("mykey", "myvalue", ttl=3600)

# Get value
value = client.get("mykey")

# Delete keys by pattern
client.delete_pattern("myprefix:*")
```

## Module-specific caching

This generic layer is designed to be extended by specific modules. For example:

- `models/cache/` - ML model result caching
- Future modules can add their own cache layers

Each module can create specialized decorators that use the generic `@cached` decorator with module-specific key functions.
