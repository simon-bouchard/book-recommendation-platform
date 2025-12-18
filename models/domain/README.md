# models/domain/README.md

# Domain Models Layer

Pure business objects representing core entities in the recommendation system.

## Purpose

Layer 1 in the architecture - provides clean, validated domain models with no external dependencies (except core constants). These models represent the business concepts and rules of the recommendation system.

## Models

### User (`user.py`)

Represents a user with their preferences and status.

```python
from models.domain import User

# Create from ORM
user = User.from_orm(orm_user)

# Check status
if user.has_preferences:
    print(f"User has {len(user.fav_subjects)} favorite subjects")

if user.is_warm:
    print("User eligible for collaborative filtering")
```

**Properties:**
- `has_preferences`: True if user has valid (non-PAD) favorite subjects
- `is_warm`: True if user exists in ALS model (faster and more robust than rating count)

### Candidate (`recommendation.py`)

Represents a book identified as potentially relevant before filtering.

```python
from models.domain import Candidate

candidate = Candidate(
    item_idx=12345,
    score=0.85,
    source="als"
)
```

**Validation:**
- Score must be non-negative
- Source must be non-empty

### RecommendedBook (`recommendation.py`)

Final recommendation with complete metadata for presentation.

```python
from models.domain import RecommendedBook

book = RecommendedBook(
    item_idx=12345,
    title="Example Book",
    score=0.85,
    num_ratings=150,
    author="Author Name",
    year=2020,
    cover_id="12345-L",
    avg_rating=4.2
)

# Convert to dict for JSON
book_dict = book.to_dict()
```

**Validation:**
- Title must be non-empty
- Score must be non-negative
- num_ratings must be non-negative
- avg_rating must be in [0, 10] if present

### RecommendationConfig (`config.py`)

Configuration for recommendation requests with validation.

```python
from models.domain import RecommendationConfig, HybridConfig

# Default configuration
config = RecommendationConfig.default(k=50, mode="auto")

# Custom configuration with diversity
config = RecommendationConfig(
    k=100,
    mode="subject",
    hybrid_config=HybridConfig(
        subject_weight=0.7,
        k_hybrid=150,
        k_subject=25,
        k_popularity=25
    )
)
```

**Validation:**
- k must be in [1, 500]
- mode must be "auto", "subject", or "behavioral"
- hybrid_config must be valid HybridConfig

### HybridConfig (`config.py`)

Configuration for hybrid candidate generation with optional pure sources.

```python
from models.domain import HybridConfig

# Default: 200 hybrid candidates only
config = HybridConfig(subject_weight=0.6)
print(config.total_candidates)  # 200

# With diversity from pure sources
config = HybridConfig(
    subject_weight=0.6,
    k_hybrid=150,       # 150 blended (subject + popularity)
    k_subject=25,       # 25 pure subject matches
    k_popularity=25     # 25 pure popular books
)
print(config.total_candidates)  # 200

# Access derived weight
print(config.popularity_weight)  # 0.4
```

**Candidate Sources:**
- `k_hybrid`: Candidates from blended score (subject_weight * sim + popularity_weight * pop)
- `k_subject`: Pure subject similarity candidates (for diversity)
- `k_popularity`: Pure popularity candidates (for diversity)

**Validation:**
- subject_weight must be in [0, 1]
- popularity_weight is automatically derived as (1 - subject_weight)
- All k values must be non-negative
- At least one k value must be > 0

**Use Cases:**
- **Default** (k_hybrid=200, others=0): Pure hybrid approach
- **With diversity** (k_hybrid=150, k_subject=25, k_popularity=25): Mix of blended and pure sources
- **Pure popularity** (k_hybrid=0, k_popularity=200): Fallback mode

## Design Principles

1. **Pure Data**: Models contain only data and local validation, no business logic
2. **Immutable**: Use dataclasses for clear, immutable structures
3. **Validated**: All models validate their inputs in `__post_init__`
4. **No Dependencies**: Only depend on core constants (PAD_IDX)
5. **Type-Safe**: Full type hints for IDE support and type checking

## Usage in Higher Layers

Domain models are used throughout the system:

- **Infrastructure Layer**: Accepts User, produces Candidates
- **Domain Logic Layer**: Transforms Candidates to RecommendedBooks
- **Service Layer**: Uses RecommendationConfig, returns RecommendedBooks
- **API Layer**: Converts between JSON and domain models
