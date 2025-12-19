# models/infrastructure/

# Infrastructure Layer

Computation primitives for the recommendation system. Provides clean, testable wrappers around ML models and similarity search with singleton + injectable pattern.

## Purpose

Layer 3 in the architecture - provides computational components that:
1. Wrap complex ML operations with clean interfaces
2. Use singleton pattern for performance (caching loaded models)
3. Support dependency injection for testing
4. Handle quality filtering and data transformations

## Components

### SimilarityIndex (`similarity_index.py`)

FAISS-based similarity search with two-pool filtering system.

**Two-Pool System Design:**

The key insight is that we want to support different requirements for queries vs results:
- **Query pool (full)**: Any book can be queried, even obscure ones with few ratings
- **Candidate pool (filtered)**: Only high-quality books appear in results

This allows users to ask "what's similar to this obscure book I love?" while ensuring recommendations are only from trusted, well-rated books.

**Example Usage:**

```python
from models.infrastructure import SimilarityIndex
from models.data.loaders import load_book_subject_embeddings, load_book_meta

# Load data
embeddings, ids = load_book_subject_embeddings()
metadata = load_book_meta()

# Create filtered index (only books with 10+ ratings as candidates)
index = SimilarityIndex.create_filtered_index(
    embeddings=embeddings,
    ids=ids,
    metadata=metadata,
    min_rating_count=10,
    normalize=True
)

# Query any book (even if it has <10 ratings)
scores, similar_ids = index.search(query_item_id=12345, k=20)

# Check status
print(f"Total items: {index.num_total}")
print(f"Candidate items: {index.num_candidates}")
print(f"Can query book 999: {index.has_item(999)}")
print(f"Can book 999 appear in results: {index.is_candidate(999)}")
```

**Quality Thresholds (from addendum):**
- ALS mode: 10+ ratings minimum
- Hybrid mode: 5+ ratings minimum
- Subject mode: No filtering

**Manual Construction:**

```python
# Build candidate mask manually
rating_counts = metadata.loc[ids, "book_num_ratings"]
candidate_mask = rating_counts >= 10

# Create index with mask
index = SimilarityIndex(
    embeddings=embeddings,
    ids=ids,
    normalize=True,
    candidate_mask=candidate_mask
)
```

**No Filtering:**

```python
# All items are both queryable and candidates
index = SimilarityIndex(
    embeddings=embeddings,
    ids=ids,
    normalize=True,
    candidate_mask=None  # or omit parameter
)
```

### SubjectEmbedder (`subject_embedder.py`)

Wrapper around attention pooling strategies for computing subject-based embeddings.

**Singleton + Injectable:**

```python
from models.infrastructure import SubjectEmbedder

# Normal usage (singleton)
embedder = SubjectEmbedder()  # Uses ATTN_STRATEGY from environment

# Embed single subject list
user_emb = embedder.embed([5, 12, 23])  # Returns (D,) array

# Embed batch
book_embs = embedder.embed_batch([
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9]
])  # Returns (N, D) array

# Check dimensionality
print(f"Embedding dim: {embedder.embedding_dim}")

# Reset singleton (for testing/reloading)
SubjectEmbedder.reset()
```

**Testing with Injection:**

```python
from unittest.mock import Mock

# Create mock pooler
mock_pooler = Mock()
mock_pooler.return_value = torch.tensor([[0.1, 0.2, 0.3]])
mock_pooler.get_embedding_dim.return_value = 3

# Inject into embedder
embedder = SubjectEmbedder(pooler=mock_pooler)

# Use normally
emb = embedder.embed([1, 2, 3])
assert mock_pooler.called
```

**Strategy Selection:**

```python
# Use specific strategy (overrides environment)
embedder = SubjectEmbedder(strategy="perdim")

# Default uses Config.get_attention_strategy()
embedder = SubjectEmbedder()  # Uses ATTN_STRATEGY env var
```

### ALSModel (`als_model.py`)

Wrapper around ALS collaborative filtering model.

**Singleton + Injectable:**

```python
from models.infrastructure import ALSModel

# Normal usage (singleton)
als = ALSModel()  # Loads from disk

# Check user/book status
if als.has_user(123):
    print("User is warm (has ALS factors)")

if als.has_book(456):
    print("Book has behavioral data")

# Generate recommendations
recommendations = als.recommend(user_id=123, k=20)
# Returns list of item_idx sorted by predicted score

# Get latent factors
user_factors = als.get_user_factors(user_id=123)  # (D,) array or None
book_factors = als.get_book_factors(item_idx=456)  # (D,) array or None

# Model info
print(f"Users: {als.num_users}")
print(f"Books: {als.num_books}")
print(f"Factors: {als.num_factors}")

# Reset singleton
ALSModel.reset()
```

**Testing with Injection:**

```python
import numpy as np

# Create mock data
user_factors = np.random.randn(10, 64)
book_factors = np.random.randn(100, 64)
user_ids = list(range(10))
book_ids = list(range(100))

# Inject into model
als = ALSModel(
    user_factors=user_factors,
    book_factors=book_factors,
    user_ids=user_ids,
    book_ids=book_ids
)

# Use normally
assert als.has_user(5)
assert als.has_book(42)
recs = als.recommend(user_id=5, k=10)
```

## Design Patterns

### Singleton + Injectable

All infrastructure components use the same pattern:

```python
class Component:
    _instance = None

    def __new__(cls, injected_data=None):
        if injected_data is not None:
            # Injection mode - bypass singleton
            instance = super().__new__(cls)
            instance._initialized = False
            return instance

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self, injected_data=None):
        if self._initialized:
            return

        self._initialized = True
        # ... initialization logic
```

**Benefits:**
- Performance: Models loaded once and cached
- Testable: Can inject mocks for unit tests
- Clean: No global state, explicit dependencies
- Memory efficient: Singleton prevents duplicate loading

### Quality Filtering

The two-pool system ensures quality without sacrificing coverage:

```python
# Problem: User wants to query obscure book
query_book_id = 99999  # Has only 2 ratings

# Traditional approach (fails):
filtered_index = build_index(books_with_10_plus_ratings)
result = filtered_index.search(query_book_id)  # ERROR: not in index

# Two-pool approach (works):
index = SimilarityIndex(
    embeddings=all_embeddings,  # Includes query_book_id
    ids=all_ids,
    candidate_mask=mask_10_plus_ratings  # Results filtered
)
scores, results = index.search(query_book_id, k=20)
# Returns 20 high-quality similar books
```

## Testing

All infrastructure components are designed for easy testing:

```python
import pytest
from models.infrastructure import SimilarityIndex, SubjectEmbedder, ALSModel

def test_similarity_index_filters_candidates():
    """Verify two-pool filtering works correctly."""
    embeddings = np.random.randn(100, 64)
    ids = list(range(100))

    # Only first 50 are candidates
    candidate_mask = np.array([True] * 50 + [False] * 50)

    index = SimilarityIndex(embeddings, ids, candidate_mask=candidate_mask)

    # Query from filtered pool (item 75)
    scores, results = index.search(75, k=10)

    # All results should be from first 50
    assert all(r < 50 for r in results)

def test_subject_embedder_with_mock():
    """Verify embedder works with injected pooler."""
    mock_pooler = Mock(return_value=torch.ones(1, 64))
    embedder = SubjectEmbedder(pooler=mock_pooler)

    emb = embedder.embed([1, 2, 3])

    assert emb.shape == (64,)
    mock_pooler.assert_called_once()

def test_als_model_with_mock_data():
    """Verify ALS model works with injected factors."""
    als = ALSModel(
        user_factors=np.random.randn(10, 64),
        book_factors=np.random.randn(100, 64),
        user_ids=list(range(10)),
        book_ids=list(range(100))
    )

    assert als.has_user(5)
    recs = als.recommend(5, k=10)
    assert len(recs) <= 10
```

## Usage in Higher Layers

Infrastructure components are used by domain logic:

**Domain Layer (Layer 4):**
```python
# Generators use infrastructure
class ALSBasedGenerator:
    def __init__(self):
        self.als = ALSModel()

    def generate(self, user, k):
        if not self.als.has_user(user.user_id):
            return []
        return self.als.recommend(user.user_id, k)
```

**Service Layer (Layer 5):**
```python
# Services orchestrate infrastructure
class SimilarityService:
    def __init__(self):
        self.embedder = SubjectEmbedder()
        # Build indices on demand

    def get_similar(self, item_idx, mode, k):
        # Use appropriate index based on mode
        ...
```

## Performance Considerations

1. **Singleton caching**: Models loaded once at startup
2. **FAISS efficiency**: Inner product search is O(ND) for N candidates
3. **Batch operations**: Use `embed_batch()` instead of looping `embed()`
4. **Memory usage**: Full embeddings kept in memory for query pool

## Memory Footprint

Typical memory usage (for reference):
- Subject embeddings: ~50K books × 64D × 4 bytes = ~12 MB
- ALS factors: ~10K users × 64D + ~30K books × 64D = ~10 MB
- FAISS index: Similar to embedding matrix size
- Total: ~25-30 MB for infrastructure layer
