# routes/models.py

REST API endpoints for ML model services (recommendations and book similarity).

## Overview

This module provides clean HTTP endpoints for the recommendation system's core ML capabilities:
- **Personalized Recommendations**: Generate book recommendations for users
- **Book Similarity**: Find similar books using various similarity modes

These endpoints use the refactored Layer 5 (Services) architecture, providing type-safe, well-documented, and performant access to ML models.

## Endpoints

### GET /profile/recommend

Generate personalized book recommendations for a user.

**Query Parameters:**
- `user` (required): User ID or username
- `_id` (default: true): If true, `user` param is user_id; if false, username
- `top_n` (default: 200, range: 1-500): Number of recommendations
- `mode` (default: "auto"): Recommendation strategy
  - `auto`: Automatically select (warm → ALS, cold → hybrid)
  - `subject`: Force subject-based recommendations
  - `behavioral`: Force ALS-based recommendations
- `w` (default: 0.6, range: 0-1): Subject weight for hybrid mode

**Example Request:**
```bash
# Get recommendations for user ID 123
GET /profile/recommend?user=123&top_n=20

# Get recommendations for username "alice"
GET /profile/recommend?user=alice&_id=false&top_n=50

# Force subject-based recommendations
GET /profile/recommend?user=123&mode=subject&w=0.8
```

**Example Response:**
```json
{
  "recommendations": [
    {
      "item_idx": 1234,
      "title": "The Great Gatsby",
      "score": 0.92,
      "num_ratings": 1500,
      "author": "F. Scott Fitzgerald",
      "year": 1925,
      "isbn": "978-0-123456-78-9",
      "cover_id": "abc123",
      "avg_rating": 4.5
    },
    ...
  ],
  "count": 20,
  "user_id": 123,
  "mode": "auto",
  "is_warm": true,
  "has_preferences": true
}
```

**Response Fields:**
- `recommendations`: List of recommended books
  - `item_idx`: Book identifier
  - `title`: Book title
  - `score`: Recommendation score (higher = more relevant)
  - `num_ratings`: Number of user ratings
  - `author`: Author name (nullable)
  - `year`: Publication year (nullable)
  - `isbn`: ISBN identifier (nullable)
  - `cover_id`: OpenLibrary cover ID (nullable)
  - `avg_rating`: Average user rating (nullable)
- `count`: Number of recommendations returned
- `user_id`: User ID recommendations were generated for
- `mode`: Actual recommendation mode used
- `is_warm`: Whether user has ALS factors (sufficient history)
- `has_preferences`: Whether user has subject preferences

**Business Logic:**

Mode selection when `mode="auto"`:
1. **Warm user** (has 10+ ratings): Use ALS collaborative filtering
2. **Cold user with preferences**: Use hybrid (subject similarity + popularity)
3. **Cold user without preferences**: Use popularity fallback

**Error Responses:**
- `404`: User not found
- `422`: Invalid parameters (e.g., mode not in allowed values)
- `500`: Internal server error

---

### GET /book/{item_idx}/similar

Find similar books using subject-based, collaborative filtering, or hybrid similarity.

**Path Parameters:**
- `item_idx` (required): Book ID to find similar books for

**Query Parameters:**
- `mode` (default: "subject"): Similarity algorithm
  - `subject`: Semantic similarity based on book subjects (no filtering)
  - `als`: Collaborative filtering based on user behavior (10+ ratings filter)
  - `hybrid`: Blended subject + ALS scores (5+ ratings filter)
- `alpha` (default: 0.6, range: 0-1): Blend weight for hybrid mode
  - 0.0 = Pure subject similarity
  - 1.0 = Pure ALS similarity
- `top_k` (default: 200, range: 1-500): Number of similar books
- `min_rating_count` (optional): Override default rating threshold
- `filter_candidates` (default: true): Enable quality filtering

**Example Request:**
```bash
# Subject-based similarity (no filtering)
GET /book/1234/similar?mode=subject&top_k=10

# Collaborative filtering (10+ ratings)
GET /book/1234/similar?mode=als&top_k=20

# Hybrid with custom blend
GET /book/1234/similar?mode=hybrid&alpha=0.7&top_k=15

# Disable filtering
GET /book/1234/similar?mode=als&filter_candidates=false

# Custom rating threshold
GET /book/1234/similar?mode=hybrid&min_rating_count=20
```

**Example Response:**
```json
{
  "similar_books": [
    {
      "item_idx": 5678,
      "title": "Tender Is the Night",
      "score": 0.89,
      "author": "F. Scott Fitzgerald",
      "year": 1934,
      "isbn": "978-0-987654-32-1",
      "cover_id": "xyz789"
    },
    ...
  ],
  "count": 10,
  "mode": "hybrid",
  "query_item_idx": 1234
}
```

**Response Fields:**
- `similar_books`: List of similar books
  - `item_idx`: Book identifier
  - `title`: Book title
  - `score`: Similarity score (higher = more similar)
  - `author`: Author name (nullable)
  - `year`: Publication year (nullable)
  - `isbn`: ISBN identifier (nullable)
  - `cover_id`: OpenLibrary cover ID (nullable)
- `count`: Number of similar books returned
- `mode`: Similarity mode used
- `query_item_idx`: Original book queried

**Filtering Behavior:**

| Mode | Default Filter | Override Parameter |
|------|---------------|-------------------|
| subject | None | N/A (no filtering) |
| als | 10+ ratings | `min_rating_count` |
| hybrid | 5+ ratings | `min_rating_count` |

**Two-Pool Architecture:**
- Query pool: ANY book can be queried (even if low-rated)
- Candidate pool: Only high-quality books appear in results

This allows users to ask "what's similar to this obscure book?" while ensuring results are only from trusted, well-rated books.

**Error Responses:**
- `422`: Invalid parameters (e.g., invalid mode or alpha out of range)
- `500`: Internal server error

## Integration

### Add to Main Application

In `main.py` or wherever you initialize your FastAPI app:

```python
from fastapi import FastAPI
from routes import api, models

app = FastAPI()

# Include main API router
app.include_router(api.router)

# Include models API router
app.include_router(models.router, tags=["models"])
```

### Update Old API (routes/api.py)

The old endpoints in `routes/api.py` can be removed or kept for backward compatibility:

**Option 1: Remove old endpoints** (recommended after migration)
```python
# Delete these from api.py:
# - @router.get("/book/{item_idx}/similar")
# - @router.get("/profile/recommend")
```

**Option 2: Keep as redirects** (for gradual migration)
```python
from routes.models import recommend_for_user as new_recommend
from routes.models import get_similar_books as new_similarity

@router.get("/book/{item_idx}/similar")
async def get_similar(item_idx: int, mode: str = "subject", alpha: float = 0.6, top_k: int = 200):
    """Deprecated: Use new models endpoint."""
    # Redirect to new endpoint or keep old implementation during transition
    return await new_similarity(item_idx, mode, alpha, top_k)
```

## Dependencies

This module depends on:
- **Layer 5 (Services)**: `RecommendationService`, `SimilarityService`
- **Layer 1 (Domain)**: `User`, `RecommendationConfig`, `HybridConfig`
- **Layer 0 (Foundation)**: `PAD_IDX`
- **Database**: SQLAlchemy ORM models
- **FastAPI**: Request/response handling

## Type Safety

All endpoints use Pydantic models for:
- **Request validation**: Query parameters are validated automatically
- **Response serialization**: Type-safe response models
- **Documentation**: Auto-generated OpenAPI/Swagger docs

Access interactive docs at: `http://localhost:8000/docs`

## Testing

### Manual Testing with cURL

**Get recommendations:**
```bash
curl "http://localhost:8000/profile/recommend?user=123&top_n=10"
```

**Get similar books:**
```bash
curl "http://localhost:8000/book/1234/similar?mode=hybrid&alpha=0.7&top_k=15"
```

### Automated Testing

Unit tests should be created in `tests/unit/routes/test_models_api.py`:

```python
def test_recommend_endpoint_returns_recommendations(test_client):
    response = test_client.get("/profile/recommend?user=123&top_n=10")

    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert data["count"] <= 10
    assert data["user_id"] == 123

def test_similarity_endpoint_returns_similar_books(test_client):
    response = test_client.get("/book/1234/similar?mode=subject&top_k=5")

    assert response.status_code == 200
    data = response.json()
    assert "similar_books" in data
    assert data["count"] <= 5
    assert data["query_item_idx"] == 1234
```

## Performance Considerations

### Caching Recommendations

Consider adding caching for frequently requested recommendations:

```python
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache

@router.get("/profile/recommend")
@cache(expire=300)  # Cache for 5 minutes
async def recommend_for_user(...):
    ...
```

### Rate Limiting

Consider rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.get("/profile/recommend")
@limiter.limit("10/minute")
async def recommend_for_user(...):
    ...
```

## Observability

### Logging

Both endpoints log structured events:
- Request start (user_id, mode, parameters)
- Request completion (count, latency)
- Errors (context, traceback)

Access logs via your logging infrastructure.

### Metrics

Consider adding Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

recommendation_requests = Counter(
    "recommendation_requests_total",
    "Total recommendation requests",
    ["mode", "user_type"]
)

recommendation_latency = Histogram(
    "recommendation_latency_seconds",
    "Recommendation request latency"
)
```

## Migration Guide

### From Old API to New API

**Before** (old recommender_strategy.py):
```python
strategy = RecommenderStrategy.get_strategy(num_ratings)
results = strategy.recommend(user_obj, db=db, top_k=top_n, w=w)
```

**After** (new recommendation_service.py):
```python
service = RecommendationService()
config = RecommendationConfig(k=top_n, mode="auto", hybrid_config=HybridConfig(subject_weight=w))
domain_user = User.from_orm(user_obj)
results = service.recommend(domain_user, config, db)
```

**Before** (old book_similarity_engine.py):
```python
strategy = get_similarity_strategy(mode=mode, alpha=alpha)
results = strategy.get_similar_books(item_idx, top_k=top_k, alpha=alpha)
```

**After** (new similarity_service.py):
```python
service = SimilarityService()
results = service.get_similar(item_idx=item_idx, mode=mode, k=top_k, alpha=alpha)
```

### Backward Compatibility

The new API maintains the same query parameters as the old API:
- ✅ Same parameter names
- ✅ Same default values
- ✅ Same response structure (with additional metadata)

Existing clients should work without changes.

## Troubleshooting

### "User not found" error

**Cause**: User ID doesn't exist in database or `_id` parameter is wrong

**Solution**:
- Verify user exists: `SELECT * FROM users WHERE user_id = 123`
- Check `_id` parameter (true for ID, false for username)

### Empty recommendations

**Cause**: User has no preferences and no interaction history

**Solution**: This is expected behavior. Service returns popularity-based recommendations.

### "No ALS data" for behavioral mode

**Cause**: User requested behavioral mode but has <10 ratings

**Solution**:
- Use `mode=auto` to automatically select appropriate strategy
- Or use `mode=subject` for subject-based recommendations

## Related Documentation

- **Service Implementation**: `models/services/recommendation_service.py`, `models/services/similarity_service.py`
- **Domain Models**: `models/domain/user.py`, `models/domain/config.py`
- **Refactor Plan**: `models/models_refactor_plan.md`
- **Service Tests**: `tests/unit/models/services/README.md`
