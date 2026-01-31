# models/cache

ML-specific caching layer for recommendation and similarity endpoints.

## Overview

This module provides optimized caching for ML model results with:
- Long TTLs for static similarity results (24 hours)
- Shorter TTLs for recommendations (1 hour)
- Automatic cache invalidation on model reload
- Smart key generation for cache sharing

## Cache Key Patterns

### Similarity Cache

All similarity results are cached for 24 hours:

```
ml:sim:subject:{item_idx}:{k}           # Subject-based similarity
ml:sim:als:{item_idx}:{k}               # ALS collaborative filtering
ml:sim:hybrid:{item_idx}:{k}:{alpha}    # Hybrid blend
```

### Recommendation Cache

Recommendations are cached for 1 hour with different strategies per mode:

```
ml:rec:behavioral:{user_id}:{top_n}              # User-specific (ALS)
ml:rec:subject:{subjects_hash}:{top_n}:{w}       # Shareable across users!
ml:rec:auto:{user_id}:{top_n}:{w}                # User-specific (decision)
```

**Key insight:** Subject-based recommendations use a hash of the subject list, allowing cache sharing between users with identical subject preferences.

## Usage

### Apply decorators to endpoints

```python
from models.cache import cached_similarity, cached_recommendations

@router.get("/book/{item_idx}/similar")
@cached_similarity
def get_similar_books(item_idx: int, mode: str, alpha: float, top_k: int):
    # Your existing code unchanged
    ...

@router.get("/profile/recommend")
@cached_recommendations
async def recommend_for_user(user: str, _id: bool, top_n: int, mode: str, w: float, db):
    # Your existing code unchanged
    ...
```

### Cache invalidation on model reload

```python
from models.cache import clear_ml_cache

@router.post("/admin/reload_models")
def reload_models_endpoint(secret: str):
    # ... existing reload logic ...

    # Clear all ML cache
    clear_ml_cache()

    return {"status": "reloaded"}
```

### Background poller integration

```python
from models.cache import clear_ml_cache

async def _reload_models(self):
    # ... existing reload logic ...

    # Clear ML cache
    clear_ml_cache()
```

## Configuration

Uses the same Redis connection as `app/cache`:

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=1              # ML cache uses DB 1 (chatbot uses DB 0)
REDIS_PASSWORD=         # Optional
```

## Cache TTLs

- **Similarity:** 24 hours (static until model reload)
- **Recommendations:** 1 hour (provides freshness without excessive computation)

## Monitoring

All cache operations are logged:

```
Cache HIT: ml:sim:subject:12345:200
Cache MISS: ml:rec:behavioral:789:50
Cache SET: ml:sim:als:12345:200 (computed in 45ms, ttl=86400s)
Cleared 1523 ML cache entries
```

## Subject Hash Strategy

Subject-based recommendations use MD5 hashing for cache keys:

```python
# User A: subjects = [5, 1, 12]
# User B: subjects = [1, 5, 12]
# Both map to same cache key (sorted before hashing)
# Result: Cache sharing between users with identical subjects!
```

This dramatically improves cache hit rates for popular subject combinations.

## Performance Impact

Expected latency improvements:
- **Similarity cache hit:** 30ms → 1-2ms (93% reduction)
- **Recommendation cache hit:** 50-80ms → 1-2ms (95-97% reduction)

Cache hit rates will vary based on:
- User behavior patterns (repeated page refreshes)
- Subject distribution (popular combinations hit more)
- Model reload frequency (clears all cache)
