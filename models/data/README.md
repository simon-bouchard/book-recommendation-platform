# models/data/

Data loading layer providing functions to load model artifacts and query databases.

## Overview

This module replaces the monolithic `ModelStore` class with focused, cacheable loading functions. It provides:

- **Loaders**: Functions to load embeddings, models, and metadata
- **Queries**: Database and DataFrame operations for recommendation pipeline
- **Caching**: Automatic caching of loaded artifacts to improve performance

## Module Structure

```
models/data/
├── __init__.py         # Public API exports
├── loaders.py          # Artifact loading functions
├── queries.py          # Database/DataFrame queries
└── README.md           # This file
```

## Usage

### Loading Embeddings

```python
from models.data import load_book_subject_embeddings, load_als_embeddings

# Load subject-based book embeddings
embeddings, book_ids = load_book_subject_embeddings()

# Load normalized version (L2-normalized)
norm_embs, book_ids = load_book_subject_embeddings(normalized=True)

# Load ALS factors
user_factors, book_factors, user_map, book_map = load_als_embeddings()

# Check if book has ALS data
from models.data.loaders import has_book_als
if has_book_als(item_idx):
    # Use ALS-based similarity
    pass
```

### Loading Models and Metadata

```python
from models.data import (
    load_bayesian_scores,
    load_book_meta,
    load_user_meta,
    load_gbt_cold_model,
    load_attention_strategy
)

# Load precomputed scores
scores = load_bayesian_scores()

# Load metadata DataFrames
book_meta = load_book_meta()  # Indexed by item_idx
user_meta = load_user_meta()  # Indexed by user_id

# Load trained models
cold_model = load_gbt_cold_model()
warm_model = load_gbt_warm_model()

# Load attention pooling strategy
strategy = load_attention_strategy("perdim")
user_emb = strategy([[1, 2, 3]])  # Pool user's favorite subjects
```

### Database Queries

```python
from models.data import (
    get_read_books,
    get_candidate_book_df,
    filter_read_books,
    add_book_embeddings
)

# Get books user has already read
read_set = get_read_books(user_id=123, db=db_session)

# Get candidate book metadata (preserves order)
candidates = [1, 5, 10, 23, 45]
df = get_candidate_book_df(candidates)

# Filter out already-read books
df = filter_read_books(df, user_id=123, db=db_session)

# Add embedding features to DataFrame
df = add_book_embeddings(df)  # Adds book_emb_0, book_emb_1, ...
```

### UI Helpers

```python
from models.data import has_book_subjects, has_book_als

# Check if book has subject embeddings (for subject similarity)
if has_book_subjects(item_idx=123):
    # Enable subject-based similarity button in UI
    pass

# Check if book has ALS factors (for behavioral similarity)
if has_book_als(item_idx=123):
    # Enable behavioral similarity button in UI
    pass

# Use in book page template context
book_data = {
    "item_idx": 123,
    "title": "Example Book",
    "has_subjects": has_book_subjects(123),
    "has_als": has_book_als(123),
}
```

### Helper Functions

```python
from models.data import (
    get_item_idx_to_row,
    compute_subject_overlap,
    decompose_embeddings,
    clean_row
)

# Map item indices to embedding rows
_, book_ids = load_book_subject_embeddings()
idx_to_row = get_item_idx_to_row(book_ids)
row = idx_to_row[item_idx]

# Compute subject overlap
overlap = compute_subject_overlap(
    user_subjects=[1, 2, 3],
    book_subjects=[2, 3, 4]
)  # Returns 2

# Decompose embeddings for features
import torch
user_emb = torch.randn(1, 64)
features = decompose_embeddings(user_emb, prefix="user_emb")
# Returns: {'user_emb_0': 0.5, 'user_emb_1': -0.3, ...}

# Clean NaN/inf values
row = {'score': float('nan'), 'rating': 4.5}
clean = clean_row(row)  # {'score': None, 'rating': 4.5}
```

## Caching

All loader functions support automatic caching via `use_cache` parameter (default `True`).

```python
# Load and cache
embeddings, ids = load_book_subject_embeddings(use_cache=True)

# Load from cache (no disk I/O)
embeddings, ids = load_book_subject_embeddings(use_cache=True)

# Force reload from disk
embeddings, ids = load_book_subject_embeddings(use_cache=False)

# Clear all cached artifacts
from models.data.loaders import clear_cache
clear_cache()

# Preload everything at startup
from models.data.loaders import preload_all_artifacts
preload_all_artifacts()
```

## API Reference

### Loaders (`loaders.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `load_book_subject_embeddings(normalized, use_cache)` | Load attention-pooled book embeddings | `(embeddings, book_ids)` |
| `load_als_embeddings(normalized, use_cache)` | Load ALS factors for users and books | `(user_factors, book_factors, user_map, book_map)` |
| `has_book_als(item_idx)` | Check if book has ALS data | `bool` |
| `load_bayesian_scores(use_cache)` | Load precomputed Bayesian scores | `np.ndarray` |
| `load_book_meta(use_cache)` | Load book metadata with Bayesian scores | `pd.DataFrame` |
| `load_user_meta(use_cache)` | Load user metadata | `pd.DataFrame` |
| `load_book_to_subjects(use_cache)` | Load book→subjects mapping | `Dict[int, List[int]]` |
| `load_gbt_cold_model(use_cache)` | Load GBT model for cold users | `LGBMRegressor` |
| `load_gbt_warm_model(use_cache)` | Load GBT model for warm users | `LGBMRegressor` |
| `load_attention_strategy(strategy, use_cache)` | Load attention pooling strategy | Strategy instance |
| `normalize_embeddings(embeddings)` | L2-normalize embedding matrix | `np.ndarray` |
| `get_item_idx_to_row(item_ids)` | Create item_idx→row mapping | `Dict[int, int]` |
| `clear_cache()` | Clear all cached artifacts | `None` |
| `preload_all_artifacts()` | Preload all artifacts into cache | `None` |

### Queries (`queries.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `get_read_books(user_id, db)` | Get books user has read | `Set[int]` |
| `get_candidate_book_df(candidate_ids)` | Get metadata for candidates (ordered) | `pd.DataFrame` |
| `filter_read_books(df, user_id, db)` | Remove already-read books | `pd.DataFrame` |
| `add_book_embeddings(df)` | Add embedding columns to DataFrame | `pd.DataFrame` |
| `compute_subject_overlap(user_subjects, book_subjects)` | Count overlapping subjects | `int` |
| `decompose_embeddings(tensor, prefix)` | Flatten tensor to dict | `Dict[str, float]` |
| `clean_row(row)` | Replace NaN/inf with None | `Dict` |
| `get_user_num_ratings(user_id)` | Get user rating count from cache | `int` |
| `has_book_subjects(item_idx)` | Check if book has subject embeddings (UI helper) | `bool` |
| `has_book_als(item_idx)` | Check if book has ALS factors (UI helper) | `bool` |

## Migration from ModelStore

The old `ModelStore` singleton has been replaced with focused functions:

```python
# OLD (ModelStore)
from models.shared_utils import ModelStore

store = ModelStore()
embeddings, ids = store.get_book_embeddings(normalized=True)
book_meta = store.get_book_meta()
strategy = store.get_attention_strategy("perdim")

# NEW (focused functions)
from models.data import (
    load_book_subject_embeddings,
    load_book_meta,
    load_attention_strategy
)

embeddings, ids = load_book_subject_embeddings(normalized=True)
book_meta = load_book_meta()
strategy = load_attention_strategy("perdim")
```

### Benefits of New Approach

1. **Clear function names**: `load_book_subject_embeddings` vs `get_book_embeddings`
2. **No singleton state**: Easier to test and reason about
3. **Explicit caching**: Cache control via function parameters
4. **Better IDE support**: Individual functions show up in autocomplete
5. **Simpler imports**: Import only what you need

## Design Principles

1. **Focused functions**: Each function does one thing well
2. **Caching by default**: Performance optimization without complexity
3. **Explicit parameters**: `normalized=True` is clearer than separate methods
4. **Type hints**: All functions have complete type annotations
5. **Error handling**: Clear error messages for missing files

## Performance Considerations

### Memory Usage

All loaders cache by default. For memory-constrained environments:

```python
# Disable caching for specific loads
embeddings, ids = load_book_subject_embeddings(use_cache=False)

# Or clear cache after use
from models.data.loaders import clear_cache
clear_cache()
```

### Startup Optimization

For production deployments, preload artifacts at startup:

```python
# In your application initialization
from models.data.loaders import preload_all_artifacts

preload_all_artifacts()  # Warms up all caches
```

### Lazy Loading

Artifacts are loaded on-demand, not at module import:

```python
# This is fast (no file I/O)
from models.data import load_book_subject_embeddings

# This loads from disk
embeddings, ids = load_book_subject_embeddings()
```

## Error Handling

All loaders raise clear exceptions:

```python
try:
    embeddings, ids = load_book_subject_embeddings()
except FileNotFoundError as e:
    print(f"Embeddings not found: {e}")

try:
    strategy = load_attention_strategy("invalid_name")
except ValueError as e:
    print(f"Invalid strategy: {e}")
```

## Testing

When testing, you can disable caching to ensure fresh data:

```python
def test_something():
    # Force reload for test isolation
    embeddings, ids = load_book_subject_embeddings(use_cache=False)
    assert len(embeddings) == len(ids)
```

## Related Documentation

- See `models/core/README.md` for path definitions
- See `models_refactor_plan.md` for overall refactor strategy
- See `naming_references.md` for artifact naming conventions
