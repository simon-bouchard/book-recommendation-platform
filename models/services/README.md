# models/services/README.md

# Services Layer

The services layer implements high-level business logic and orchestration for the recommendation system. Services coordinate between domain components (generators, filters, pipelines) and provide clean interfaces for the API layer.

## Overview

This layer contains two main services:

- **RecommendationService**: Personalized book recommendations with automatic strategy selection
- **SimilarityService**: Book-to-book similarity search with three modes (subject, ALS, hybrid)

## RecommendationService

Generates personalized recommendations by selecting the appropriate strategy based on user characteristics.

### Business Rules

The service automatically selects recommendation strategies:

| User Type | Strategy | Generator(s) |
|-----------|----------|--------------|
| Warm user (10+ ratings) | Collaborative filtering | ALS |
| Cold user with preferences | Hybrid content-based | Subject + Popularity blend |
| Cold user without preferences | Popularity fallback | Bayesian popularity |

### Usage

```python
from models.services import RecommendationService
from models.domain.user import User
from models.domain.config import RecommendationConfig

service = RecommendationService()

# Auto mode (recommended)
config = RecommendationConfig(k=20, mode="auto")
recommendations = service.recommend(user, config, db)

# Force behavioral mode
config = RecommendationConfig(k=20, mode="behavioral")
recommendations = service.recommend(user, config, db)

# Force subject mode
config = RecommendationConfig(k=20, mode="subject")
recommendations = service.recommend(user, config, db)

# Custom hybrid blend
from models.domain.config import HybridConfig
config = RecommendationConfig(
    k=20,
    mode="subject",
    hybrid_config=HybridConfig(
        subject_weight=0.7,  # 70% subject similarity
        k_hybrid=150,
        k_subject=25,
        k_popularity=25
    )
)
recommendations = service.recommend(user, config, db)
```

### Return Type

Returns `List[RecommendedBook]` with complete metadata:

```python
RecommendedBook(
    item_idx=123,
    title="The Great Gatsby",
    score=0.85,
    num_ratings=1200,
    author="F. Scott Fitzgerald",
    year=1925,
    isbn="9780743273565",
    cover_id="8267342",
    avg_rating=4.1
)
```

### Observability

The service provides structured logging for monitoring:

```python
# Start
logger.info("Recommendation started", extra={
    "user_id": 123,
    "mode": "auto",
    "is_warm": True,
    "has_preferences": True,
    "k": 20
})

# Success
logger.info("Recommendation completed", extra={
    "user_id": 123,
    "mode": "auto",
    "count": 20,
    "latency_ms": 45
})

# Error
logger.error("Recommendation failed", extra={
    "user_id": 123,
    "mode": "auto",
    "error": "..."
}, exc_info=True)
```

## SimilarityService

Finds similar books using three different similarity modes, each with appropriate quality filtering.

### Similarity Modes

| Mode | Based On | Candidate Filter | Use Case |
|------|----------|------------------|----------|
| Subject | Semantic subject similarity | None (all books) | Content-based discovery |
| ALS | Collaborative filtering patterns | 10+ ratings | Behavioral similarity |
| Hybrid | Blended subject + ALS scores | 5+ ratings | Best of both approaches |

### Two-Pool Architecture

The service implements a two-pool system for quality control:

- **Query pool**: Any book can be queried (even low-rated/obscure books)
- **Candidate pool**: Only high-quality books appear in results

This allows users to ask "what's similar to this obscure book?" while ensuring results are trustworthy.

### Usage

```python
from models.services import SimilarityService

service = SimilarityService()

# Subject-based similarity (no filtering)
results = service.get_similar(
    item_idx=123,
    mode="subject",
    k=20
)

# ALS-based similarity (10+ ratings)
results = service.get_similar(
    item_idx=123,
    mode="als",
    k=20
)

# Hybrid with custom blend
results = service.get_similar(
    item_idx=123,
    mode="hybrid",
    k=20,
    alpha=0.7  # 70% ALS, 30% subject
)

# Override rating threshold
results = service.get_similar(
    item_idx=123,
    mode="hybrid",
    k=20,
    alpha=0.6,
    min_rating_count=20  # Stricter than default 5
)

# Disable filtering (get all results)
results = service.get_similar(
    item_idx=123,
    mode="als",
    k=20,
    filter_candidates=False
)
```

### Return Type

Returns `List[Dict]` with book metadata:

```python
{
    "item_idx": 456,
    "title": "To Kill a Mockingbird",
    "author": "Harper Lee",
    "year": 1960,
    "isbn": "9780061120084",
    "cover_id": "295577",
    "score": 0.92
}
```

### Hybrid Mode Implementation

The hybrid mode implements score blending with alignment:

1. **Initialization** (one-time cost on first hybrid query):
   - Loads subject embeddings and ALS factors
   - Builds alignment mapping: subject space → ALS space
   - Handles books missing in ALS (fills zeros)

2. **Query processing**:
   - Computes subject scores for all books
   - Computes ALS scores for all books
   - Aligns ALS scores to subject space
   - Blends: `final = (1-alpha)*subject + alpha*ALS`
   - Filters by rating count (default 5+)
   - Returns top k by score

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `item_idx` | int | Required | Book to find similar items for |
| `mode` | str | "subject" | "subject", "als", or "hybrid" |
| `k` | int | 200 | Number of results to return |
| `alpha` | float | 0.6 | Hybrid blend weight (0.0=subject, 1.0=ALS) |
| `min_rating_count` | int | None | Override default threshold (10 for ALS, 5 for hybrid) |
| `filter_candidates` | bool | True | Enable/disable candidate filtering |

## Error Handling

Both services log errors with context and re-raise exceptions:

```python
try:
    recommendations = service.recommend(user, config, db)
except Exception as e:
    # Logged with user_id, mode, error message, and stack trace
    # Exception re-raised for API layer to handle
    pass
```

## Performance Considerations

### Lazy Loading

Both services use lazy initialization:
- Indices built on first use
- Cached in memory for subsequent requests
- Hybrid mode initialization ~100-200ms one-time cost

### Memory Usage

Approximate memory footprint:
- Subject index: ~50MB (embeddings + FAISS)
- ALS index: ~30MB (factors + FAISS)
- Hybrid cache: ~80MB (both embeddings + alignment maps)
- Book metadata: ~20MB (DataFrame)

Total: ~180MB when all modes used

### Latency

Typical query times:
- Subject similarity: 5-10ms
- ALS similarity: 5-10ms
- Hybrid similarity: 15-25ms (after initialization)
- Recommendation (warm user): 20-40ms
- Recommendation (cold user): 30-60ms

## Integration with API Layer

Services are designed to be called directly from FastAPI routes:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.services import RecommendationService, SimilarityService
from models.domain.user import User
from models.domain.config import RecommendationConfig

router = APIRouter()
rec_service = RecommendationService()
sim_service = SimilarityService()

@router.post("/recommend")
def recommend(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    user = User.from_orm(get_user(request.user_id, db))
    config = RecommendationConfig(k=request.k, mode=request.mode)

    recommendations = rec_service.recommend(user, config, db)

    return {
        "recommendations": [r.to_dict() for r in recommendations],
        "user_type": "warm" if user.is_warm else "cold"
    }

@router.get("/book/{item_idx}/similar")
def get_similar(
    item_idx: int,
    mode: str = "subject",
    k: int = 20,
    alpha: float = 0.6
):
    results = sim_service.get_similar(item_idx, mode, k, alpha)
    return {"similar_books": results}
```

## Testing

Key test scenarios:

```python
def test_warm_user_gets_als():
    user = User(user_id=123, fav_subjects=[1,2,3])
    # Mock ALSModel to return True for has_user(123)

    service = RecommendationService()
    config = RecommendationConfig.default()

    recommendations = service.recommend(user, config, mock_db)
    # Should use ALS pipeline

def test_cold_user_with_prefs_gets_hybrid():
    user = User(user_id=456, fav_subjects=[1,2,3])
    # Mock ALSModel to return False for has_user(456)

    service = RecommendationService()
    config = RecommendationConfig.default()

    recommendations = service.recommend(user, config, mock_db)
    # Should use hybrid pipeline

def test_hybrid_similarity_filters_low_rated():
    service = SimilarityService()

    results = service.get_similar(item_idx=123, mode="hybrid", k=10)

    # All results should have 5+ ratings
    metadata = load_book_meta()
    for result in results:
        assert metadata.loc[result["item_idx"], "book_num_ratings"] >= 5
```

## Dependencies

Services depend on:
- **Domain layer**: User, Candidate, RecommendedBook, RecommendationConfig, generators, filters, pipeline
- **Infrastructure layer**: SubjectEmbedder, ALSModel, SimilarityIndex
- **Data layer**: Loaders (embeddings, metadata, scores)
- **External**: SQLAlchemy Session (for filtering read books)

## Logging Configuration

Configure Python logging in your application startup:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set models.services to INFO or DEBUG
logging.getLogger('models.services').setLevel(logging.INFO)
```

For structured logging with JSON output, use a JSON formatter:

```python
import logging
import json_log_formatter

formatter = json_log_formatter.JSONFormatter()
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger('models.services')
logger.addHandler(handler)
```
