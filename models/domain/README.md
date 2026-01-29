# Domain Layer - Business Models and Logic

This layer contains both the pure business entities (Layer 1) and the recommendation logic (Layer 4) that operates on those entities.

## Table of Contents

- [Layer 1: Domain Models](#layer-1-domain-models)
  - [User](#user-userpy)
  - [Candidate](#candidate-recommendationpy)
  - [RecommendedBook](#recommendedbook-recommendationpy)
  - [RecommendationConfig](#recommendationconfig-configpy)
  - [HybridConfig](#hybridconfig-configpy)
  - [Design Principles](#design-principles)
- [Layer 4: Recommendation Business Logic](#layer-4-recommendation-business-logic)
  - [Architecture](#architecture)
  - [Components](#components)
  - [Usage Examples](#usage-examples)
  - [Design Decisions](#design-decisions)
  - [Testing Strategy](#testing-strategy)
  - [Dependencies](#dependencies)
- [Usage Across Layers](#usage-across-layers)

---

## Layer 1: Domain Models

Pure business objects representing core entities in the recommendation system.

### Purpose

Provides clean, validated domain models with no external dependencies (except core constants). These models represent the business concepts and rules of the recommendation system.

### Models

#### User (`user.py`)

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

#### Candidate (`recommendation.py`)

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

#### RecommendedBook (`recommendation.py`)

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

#### RecommendationConfig (`config.py`)

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

#### HybridConfig (`config.py`)

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

### Design Principles

1. **Pure Data**: Models contain only data and local validation, no business logic
2. **Immutable**: Use dataclasses for clear, immutable structures
3. **Validated**: All models validate their inputs in `__post_init__`
4. **No Dependencies**: Only depend on core constants (PAD_IDX)
5. **Type-Safe**: Full type hints for IDE support and type checking

---

## Layer 4: Recommendation Business Logic

Core business logic for generating book recommendations. Implements a composable pipeline architecture that separates concerns and enables flexible recommendation strategies.

### Architecture

The domain layer follows a pipeline pattern with four main components:

```
User → Generator → Filter → Ranker → Recommendations
          ↓
      Fallback (if empty)
```

### Components

#### 1. Candidate Generators (`candidate_generation.py`)

Generate initial candidate lists based on different relevance signals:

- **SubjectBasedGenerator**: Finds books similar to user's favorite subjects using attention-pooled embeddings
- **ALSBasedGenerator**: Uses collaborative filtering (ALS matrix factorization) for warm users
- **BayesianPopularityGenerator**: Returns quality-weighted popular books (fallback for cold users)
- **HybridGenerator**: Blends multiple generators with weighted scoring

**Design principle**: Generators focus purely on relevance. They don't filter by quality or user history.

#### 2. Filters (`filters.py`)

Apply quality control and exclusions to candidates:

- **ReadBooksFilter**: Removes books user has already interacted with
- **MinRatingCountFilter**: Filters out books with insufficient ratings
- **FilterChain**: Composes multiple filters into a single filter
- **NoFilter**: Pass-through (no filtering)

**Design principle**: Filters apply constraints. Composable via FilterChain for flexibility.

#### 3. Rankers (`rankers.py`)

Determine final ordering of candidates:

- **NoOpRanker**: Preserves generator ordering (assumes generators pre-sort by score)
- **ScoreRanker**: Sorts by candidate score (useful when combining multiple sources)

**Design principle**: Rankers handle final ordering. Most generators already sort, so NoOpRanker is common.

#### 4. Pipeline (`pipeline.py`)

Orchestrates the complete recommendation flow:

```python
pipeline = RecommendationPipeline(
    generator=ALSBasedGenerator(),
    fallback_generator=BayesianPopularityGenerator(),
    filter=ReadBooksFilter(),
    ranker=NoOpRanker()
)

recommendations = pipeline.recommend(user, k=20, db=session)
```

**Flow**:
1. Generate candidates from primary generator (with 2x k buffer for filtering)
2. If empty, use fallback generator
3. Apply filter (e.g., remove read books)
4. Rank candidates
5. Return top k

## Usage Examples

### Basic ALS Recommendations (Warm Users)

```python
from models.domain.pipeline import RecommendationPipeline
from models.domain.candidate_generation import ALSBasedGenerator, BayesianPopularityGenerator
from models.domain.filters import ReadBooksFilter
from models.domain.rankers import NoOpRanker

pipeline = RecommendationPipeline(
    generator=ALSBasedGenerator(),
    fallback_generator=BayesianPopularityGenerator(),  # If user not in ALS
    filter=ReadBooksFilter(),
    ranker=NoOpRanker()
)

recommendations = pipeline.recommend(user, k=20, db=db_session)
```

### Hybrid Recommendations (Cold Users)

```python
from models.domain.candidate_generation import (
    HybridGenerator,
    SubjectBasedGenerator,
    BayesianPopularityGenerator
)

# Blend subject similarity (60%) + popularity (40%)
hybrid = HybridGenerator([
    (SubjectBasedGenerator(), 0.6),
    (BayesianPopularityGenerator(), 0.4)
])

pipeline = RecommendationPipeline(
    generator=hybrid,
    filter=ReadBooksFilter(),
    ranker=NoOpRanker()
)

recommendations = pipeline.recommend(user, k=20, db=db_session)
```

### Multiple Filters

```python
from models.domain.filters import FilterChain, ReadBooksFilter, MinRatingCountFilter

# Only recommend books with 10+ ratings that user hasn't read
filter_chain = FilterChain([
    ReadBooksFilter(),
    MinRatingCountFilter(min_count=10)
])

pipeline = RecommendationPipeline(
    generator=SubjectBasedGenerator(),
    filter=filter_chain,
    ranker=NoOpRanker()
)
```

## Design Decisions

### Why separate generators from filters?

**Generators** focus on relevance - "what might this user like?"
**Filters** apply constraints - "what can we actually recommend?"

This separation:
- Keeps generators testable (no DB dependencies for most)
- Makes quality thresholds configurable (different filters for different contexts)
- Allows reusing same generator with different filtering rules

### Why allow empty candidate lists?

Generators can return `[]` for invalid inputs (e.g., SubjectBasedGenerator with no preferences). The pipeline handles this gracefully via fallback generators. This is cleaner than passing mode flags through multiple layers.

### Why NoOpRanker as default?

Most generators already return candidates sorted by score. Re-ranking is only needed when:
- Combining candidates from multiple unordered sources
- Applying a different scoring function post-generation

NoOpRanker avoids unnecessary sorting when candidates are already in the right order.

## Testing Strategy

Each component is independently testable:

```python
# Test generator in isolation
def test_subject_generator_returns_empty_without_preferences():
    generator = SubjectBasedGenerator()
    user = User(user_id=1, fav_subjects=[PAD_IDX])

    candidates = generator.generate(user, k=10)

    assert candidates == []

# Test filter with mock candidates
def test_read_books_filter():
    filter = ReadBooksFilter()
    candidates = [Candidate(1, 0.9, "test"), Candidate(2, 0.8, "test")]
    # ... mock DB to return item_idx=1 as read

    filtered = filter.apply(candidates, user, db)

    assert len(filtered) == 1
    assert filtered[0].item_idx == 2

# Test pipeline with mock components
def test_pipeline_uses_fallback():
    primary = Mock(return_value=[])  # Returns empty
    fallback = Mock(return_value=[Candidate(1, 0.9, "fallback")])

    pipeline = RecommendationPipeline(primary, fallback)
    results = pipeline.recommend(user, k=10)

    assert len(results) > 0
    fallback.assert_called_once()
```

## Dependencies

- **Infrastructure layer**: SubjectEmbedder, ALSModel, SimilarityIndex
- **Data layer**: load_book_subject_embeddings, load_bayesian_scores, queries
- **Domain models**: User, Candidate

No dependencies on service or API layers (domain is at the core).

---

## Usage Across Layers

Domain models are used throughout the system:

- **Infrastructure Layer** (Layer 3): Accepts User, produces raw scores
- **Domain Logic Layer** (Layer 4): Uses generators/filters/pipeline to transform User → Candidates
- **Service Layer** (Layer 5): Uses RecommendationConfig, orchestrates pipeline, returns RecommendedBooks
- **API Layer** (Layer 6): Converts between JSON and domain models
