# tests/unit/models/domain/README.md
"""
Unit tests for the recommendation system's domain layer (Layer 4).

Tests candidate generation, filtering, ranking, and pipeline orchestration
with comprehensive mocking to ensure fast, deterministic execution.
"""

## Overview

This directory contains unit tests for Layer 4 (Domain Logic) of the recommendation system refactor:

- **Candidate Generation**: Various strategies for generating book recommendations
- **Filtering**: Removing unwanted candidates (e.g., already-read books)
- **Ranking**: Sorting candidates by relevance
- **Pipeline**: Orchestrating the complete recommendation flow

All tests use dependency injection and mocking to avoid external dependencies (database, file I/O, model loading).

## Test Files

### test_candidate_generation.py (117 tests)

Tests all candidate generation strategies:

**SubjectBasedGenerator** (24 tests)
- Interface compliance
- Embedding computation and similarity search
- Empty result handling for users without preferences
- Score transformation (cosine similarity → non-negative)
- Edge cases (k=0, k > catalog size)

**ALSBasedGenerator** (21 tests)
- Collaborative filtering recommendations
- Cold user handling (returns empty)
- Factor computation and ranking
- Edge cases

**BayesianPopularityGenerator** (11 tests)
- Popularity-based fallback
- Always succeeds (no user dependencies)
- Bayesian score ranking
- Edge cases

**HybridGenerator** (26 tests)
- Multi-source blending with weighted averaging
- Score normalization and deduplication
- Weight validation (must be positive, sum handled automatically)
- Empty source handling

**Configuration & Edge Cases** (35 tests)
- Constructor validation
- k=0 handling across all generators
- Large k handling
- Empty catalog scenarios

### test_filters.py (19 tests)

Tests candidate filtering strategies:

**ReadBooksFilter** (10 tests)
- Database queries for user interactions
- Filtering logic (removes read/rated books)
- Empty candidate handling
- Database session validation (requires non-None db)

**MinRatingCountFilter** (6 tests)
- Rating count thresholds
- Metadata-based filtering
- Constructor validation (min_count >= 0)

**NoFilter** (3 tests)
- Pass-through behavior
- Interface compliance

### test_pipeline.py (23 tests)

Tests the recommendation pipeline orchestration:

**Component Integration** (8 tests)
- Generator → Filter → Ranker flow
- Fallback generator triggering (only when primary returns empty initially)
- Default components (NoFilter, NoOpRanker when None passed)

**Parameter Handling** (5 tests)
- Buffer sizing (requests k*2 or 500, whichever is larger)
- Database session passing
- k parameter validation

**Edge Cases** (10 tests)
- Empty results at each stage
- Filter removes all candidates (returns empty, no fallback)
- All components return empty
- k=0 handling

### test_rankers.py (8 tests)

Tests candidate ranking strategies:

**NoOpRanker** (3 tests)
- Pass-through behavior (preserves generator order)
- Interface compliance

**ScoreRanker** (4 tests)
- Descending score sorting
- Tie handling
- Empty candidate handling

**Validation** (1 test)
- Candidate enforces non-negative scores

## Running Tests

### Run all domain layer tests:
```bash
pytest tests/unit/models/domain/ -v
```

### Run specific test file:
```bash
pytest tests/unit/models/domain/test_candidate_generation.py -v
```

### Run specific test class:
```bash
pytest tests/unit/models/domain/test_candidate_generation.py::TestSubjectBasedGenerator -v
```

### Run specific test:
```bash
pytest tests/unit/models/domain/test_candidate_generation.py::TestSubjectBasedGenerator::test_generate_returns_list_of_candidates -v
```

### Run with coverage:
```bash
pytest tests/unit/models/domain/ --cov=models.domain --cov-report=html
```

### Run in watch mode (requires pytest-watch):
```bash
ptw tests/unit/models/domain/
```

## Test Structure

All tests follow consistent patterns:

### 1. Fixtures (Dependency Injection)

Tests use pytest fixtures to inject mock dependencies:

```python
@pytest.fixture
def mock_embedder():
    """Mock SubjectEmbedder with consistent return values."""
    embedder = Mock()
    embedder.embed.return_value = np.random.randn(16).astype(np.float32)
    return embedder

@pytest.fixture
def mock_user():
    """Standard user with preferences."""
    return User(user_id=123, fav_subjects=[5, 12, 23])
```

### 2. Monkeypatching for Loader Functions

Tests monkeypatch module-level loader functions to avoid file I/O:

```python
def test_something(monkeypatch):
    monkeypatch.setattr(
        "models.domain.candidate_generation.load_bayesian_scores",
        lambda **kwargs: np.linspace(0.9, 0.1, 100)
    )
```

**Important**: Use `**kwargs` in lambdas because loaders are called with keyword arguments.

### 3. Arrange-Act-Assert Pattern

```python
def test_generate_returns_candidates(mock_embedder, mock_similarity_index, mock_user):
    # Arrange
    generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)

    # Act
    candidates = generator.generate(mock_user, k=10)

    # Assert
    assert len(candidates) == 10
    assert all(isinstance(c, Candidate) for c in candidates)
```

## Key Testing Patterns

### Dependency Injection

All classes accept optional constructor parameters for testing:

```python
# Production: uses singletons
generator = SubjectBasedGenerator()

# Testing: injects mocks
generator = SubjectBasedGenerator(mock_embedder, mock_similarity_index)
```

### Mock Attribute Requirements

When mocking infrastructure components, ensure all required attributes exist:

```python
# SubjectEmbedder mock needs:
mock_embedder.embed.return_value = np.array([...])

# SimilarityIndex mock needs:
mock_index.embeddings_full = np.random.randn(100, 16)
mock_index.ids_full = np.arange(1000, 1100)

# ALSModel mock needs:
mock_als.has_user.return_value = True
mock_als.book_factors = np.random.randn(100, 64)
mock_als.book_row_to_id = {i: i+1000 for i in range(100)}
```

### Score Transformation Testing

Candidate scores must be non-negative. Generators transform raw similarities:

```python
# SubjectBasedGenerator transforms cosine similarity [-1, 1] to [0, 1]:
scores = (cosine_scores + 1) / 2

# Tests verify this happens:
candidates = generator.generate(user, k=10)
assert all(c.score >= 0 for c in candidates)  # No negative scores
```

## Common Pitfalls

### 1. Lambda Signatures in Monkeypatch

**Wrong**: Positional arguments
```python
lambda use_cache: np.array([...])  # FAILS - missing positional arg
```

**Correct**: Keyword arguments
```python
lambda **kwargs: np.array([...])  # WORKS - accepts any kwargs
```

### 2. Mock Attribute Shape Mismatches

**Wrong**: 1D embeddings when 2D expected
```python
mock_embedder.embed.return_value = np.array([1, 2, 3])  # Wrong shape
```

**Correct**: Proper shape
```python
mock_embedder.embed.return_value = np.random.randn(16).astype(np.float32)
```

### 3. Forgetting Mock Attributes

**Wrong**: Missing required attributes
```python
mock_index = Mock()
# Missing embeddings_full and ids_full!
```

**Correct**: All attributes defined
```python
mock_index = Mock()
mock_index.embeddings_full = np.random.randn(100, 16)
mock_index.ids_full = np.arange(1000, 1100)
```

### 4. Set Ordering in Assertions

**Wrong**: Comparing sets directly
```python
assert set(candidates) == set(expected)  # Fails - sets are unordered
```

**Correct**: Compare sorted or use containment
```python
assert sorted(candidates) == sorted(expected)
# OR
assert all(c in expected for c in candidates)
```

## Test Coverage

Current coverage: **~95%** of domain layer code

Uncovered areas:
- Error paths in rarely-triggered edge cases
- Some logging statements
- Import-time errors

## Performance

All tests run in **< 2 seconds** total:
- No file I/O (all mocked)
- No database queries (mocked or in-memory)
- No model loading (mocked)
- Deterministic random data (numpy with fixed seeds where needed)

## Debugging Failed Tests

### 1. Check Mock Setup

```bash
pytest tests/unit/models/domain/test_candidate_generation.py::TestSubjectBasedGenerator::test_generate_returns_list_of_candidates -vv --tb=short
```

Look for:
- `TypeError: missing required positional argument` → Lambda signature issue
- `AttributeError: Mock object has no attribute` → Missing mock attribute
- `ValueError: Score must be non-negative` → Real bug in generator (needs score transformation)

### 2. Inspect Mock Calls

```python
# In test:
generator.generate(user, k=10)
mock_embedder.embed.assert_called_once_with([5, 12, 23])  # Check what was called
```

### 3. Print Intermediate Values

```python
# Add debug prints:
candidates = generator.generate(user, k=10)
print(f"Got {len(candidates)} candidates")
print(f"Scores: {[c.score for c in candidates]}")
```

## Integration with CI/CD

These tests run on every commit:

```yaml
# .github/workflows/test.yml
- name: Run domain layer tests
  run: |
    pytest tests/unit/models/domain/ \
      --cov=models.domain \
      --cov-fail-under=90 \
      --junit-xml=test-results/domain.xml
```

## Future Improvements

1. **Property-based testing**: Use `hypothesis` for generators
2. **Snapshot testing**: Compare full candidate outputs
3. **Performance benchmarks**: Track test execution time
4. **Mutation testing**: Verify test quality with `mutmut`

## Related Documentation

- **Refactor Plan**: `/mnt/project/models_refactor_plan.md`
- **Domain Models**: `/mnt/project/domain/`
- **Infrastructure Layer**: `/mnt/project/infrastructure/`
- **Service Layer Tests**: `tests/unit/models/services/README.md` (coming next)

## Questions?

If tests fail unexpectedly:
1. Check mock setup (fixtures)
2. Verify lambda signatures (`**kwargs`)
3. Ensure all mock attributes exist
4. Check score transformation (non-negative requirement)
5. Look for Python module caching (restart pytest)

For implementation questions, refer to the refactor plan or the actual domain layer code in `/mnt/project/domain/`.
