# Infrastructure Layer Unit Tests

Comprehensive test suite for Layer 3 (Infrastructure) components: ALSModel, SimilarityIndex, and SubjectEmbedder.

## Overview

These tests validate the computational primitives that power the recommendation system. All infrastructure components use a **singleton + injectable** pattern that allows for both performance (caching) in production and testability (mocking) in tests.

## Test Philosophy

### Dependency Injection Over File Loading

All tests use **mock data injection** instead of loading real model files:

```python
# ✅ GOOD: Fast, deterministic, isolated
mock_data = create_mock_als_data(n_users=10, n_books=50)
als = ALSModel(
    user_factors=mock_data["user_factors"],
    book_factors=mock_data["book_factors"],
    user_ids=mock_data["user_ids"],
    book_ids=mock_data["book_ids"]
)

# ❌ BAD: Slow, requires files, environment-dependent
als = ALSModel()  # Loads from disk
```

**Benefits**:
- ⚡ **Fast**: No disk I/O (tests run in ~5 seconds)
- 🎯 **Deterministic**: Seeded random data produces identical results
- 🔒 **Isolated**: No dependency on file system or trained models
- 🌍 **Portable**: Can run in any environment (CI, local, containers)

### Mock Data Characteristics

- **Small enough** for fast execution (10-50 items typical)
- **Large enough** to test edge cases (empty results, k > catalog size)
- **Seeded** for reproducibility (`np.random.seed(42)`)
- **Covers all code paths** (users with/without data, books with/without factors)

## Test Files

### test_als_model.py (13KB, 92 tests)

Tests for ALS collaborative filtering wrapper.

**Test Classes**:
- `TestALSModelInitialization` - Singleton pattern, injection bypass, data storage
- `TestHasUserHasBook` - Membership checks, type coercion
- `TestRecommend` - Recommendation generation, k parameter validation
- `TestGetFactors` - Factor retrieval for users and books
- `TestProperties` - Accessor properties (num_users, num_books, num_factors)
- `TestResetSingleton` - Singleton lifecycle management
- `TestEdgeCases` - Empty models, k=0, negative k, missing entities

**Key Patterns**:
```python
# Create mock ALS data
@pytest.fixture
def mock_als_data():
    np.random.seed(42)
    return {
        "user_factors": np.random.randn(10, 16).astype(np.float32),
        "book_factors": np.random.randn(50, 16).astype(np.float32),
        "user_ids": list(range(100, 110)),
        "book_ids": list(range(1000, 1050))
    }

# Inject into ALSModel
@pytest.fixture
def als_model(mock_als_data):
    model = ALSModel(**mock_als_data)
    yield model
    ALSModel.reset()  # Clean up singleton
```

**Critical Tests**:
- Validates bidirectional mappings (user_id ↔ row, book_id ↔ row)
- Tests determinism (same query → same results)
- Tests k validation (k ≤ 0 returns empty list)
- Tests missing user/book (returns empty list or None)

### test_similarity_index.py (17KB, 89 tests)

Tests for FAISS-based similarity with two-pool filtering system.

**Test Classes**:
- `TestSimilarityIndexInitialization` - Index creation, normalization, validation
- `TestTwoPoolSystem` - Query pool vs candidate pool separation
- `TestSearch` - FAISS search, k parameter, query exclusion
- `TestCreateFilteredIndex` - Factory method, rating threshold filtering
- `TestProperties` - num_total, num_candidates accessors
- `TestEdgeCases` - Empty pools, single candidate, identical embeddings
- `TestRealWorldScenario` - ALS mode (10+), hybrid (5+), subject (no filter)

**Key Patterns**:
```python
# Create mock embeddings
@pytest.fixture
def mock_embeddings():
    np.random.seed(42)
    embeddings = np.random.randn(50, 16).astype(np.float32)
    ids = list(range(1000, 1050))
    return embeddings, ids

# Test two-pool system
def test_two_pool_query_any_item(mock_embeddings):
    embeddings, ids = mock_embeddings

    # Only items with even IDs are candidates
    mask = np.array([i % 2 == 0 for i in ids])
    index = SimilarityIndex(embeddings, ids, candidate_mask=mask)

    # Can query ANY item (even odd IDs)
    scores, results = index.search(query_item_id=1001, k=5)

    # But results ONLY from even IDs (candidates)
    assert all(item_id % 2 == 0 for item_id in results)
```

**Critical Tests**:
- Validates two-pool system (query any item, results from candidates only)
- Tests L2 normalization (unit vectors)
- Tests factory method with rating thresholds
- Tests metadata handling (missing items use reindex + fillna(0))
- Tests real-world filtering scenarios (ALS 10+, hybrid 5+, subject no-filter)

### test_subject_embedder.py (14KB, 73 tests)

Tests for attention pooling wrapper.

**Test Classes**:
- `TestSubjectEmbedderInitialization` - Singleton pattern, injection bypass
- `TestEmbed` - Single embedding computation
- `TestEmbedBatch` - Batch embedding computation
- `TestEmbeddingDim` - Dimension property
- `TestResetSingleton` - Singleton lifecycle
- `TestEdgeCases` - Large batches, empty subjects, duplicates
- `TestNoGradientTracking` - torch.no_grad context validation
- `TestIntegrationWithMockPooler` - Realistic usage patterns

**Key Patterns**:
```python
# Create mock pooler
@pytest.fixture
def mock_pooler():
    pooler = Mock()

    def mock_forward(subjects_list):
        emb_dim = 16

        if len(subjects_list) == 0:  # Handle empty batch
            return torch.empty(0, emb_dim)

        embeddings = []
        for subjects in subjects_list:
            val = sum(subjects) if subjects else 0
            emb = torch.full((emb_dim,), float(val) / 10.0)
            embeddings.append(emb)

        return torch.stack(embeddings)

    pooler.side_effect = mock_forward
    pooler.get_embedding_dim.return_value = 16
    return pooler

# Inject into SubjectEmbedder
@pytest.fixture
def subject_embedder(mock_pooler):
    embedder = SubjectEmbedder(pooler=mock_pooler)
    yield embedder
    SubjectEmbedder.reset()
```

**Critical Tests**:
- Tests singleton bypass with injection
- Tests embed() and embed_batch() consistency
- Tests no gradient tracking (torch.no_grad)
- Tests empty batch handling (returns torch.empty(0, dim))
- Tests realistic usage patterns (recommendation pipeline, precompute scripts)

## Running the Tests

### Run All Infrastructure Tests
```bash
pytest tests/unit/models/infrastructure/ -v
```

### Run Specific Test File
```bash
pytest tests/unit/models/infrastructure/test_als_model.py -v
pytest tests/unit/models/infrastructure/test_similarity_index.py -v
pytest tests/unit/models/infrastructure/test_subject_embedder.py -v
```

### Run Specific Test Class
```bash
pytest tests/unit/models/infrastructure/test_als_model.py::TestRecommend -v
```

### Run Specific Test
```bash
pytest tests/unit/models/infrastructure/test_als_model.py::TestRecommend::test_returns_correct_number_of_recommendations -v
```

### Run with Coverage
```bash
pytest tests/unit/models/infrastructure/ --cov=models.infrastructure --cov-report=html
```

### Clear Cache (if tests fail unexpectedly)
```bash
pytest --cache-clear tests/unit/models/infrastructure/
```

## Expected Results

All tests should pass:
```
tests/unit/models/infrastructure/test_als_model.py ................ [ 38%]
tests/unit/models/infrastructure/test_similarity_index.py ......... [ 73%]
tests/unit/models/infrastructure/test_subject_embedder.py ......... [100%]

===================== 254 passed in 5.74s =====================
```

## Test Organization

```
tests/unit/models/infrastructure/
├── __init__.py
├── test_als_model.py          # ALS wrapper tests
├── test_similarity_index.py   # FAISS similarity tests
└── test_subject_embedder.py   # Attention pooling tests
```

## Common Patterns

### Singleton Reset in Fixtures

Always reset singletons after tests to avoid pollution:

```python
@pytest.fixture
def als_model(mock_als_data):
    model = ALSModel(**mock_als_data)
    yield model
    ALSModel.reset()  # ← Critical cleanup
```

### Type Coercion Testing

Test that components handle type variations gracefully:

```python
def test_has_user_with_string_id(als_model):
    """Should coerce string to int."""
    assert als_model.has_user("100") == True
```

### Determinism Testing

Verify that same inputs produce same outputs:

```python
def test_recommend_is_deterministic(als_model):
    """Same query should produce same results."""
    recs1 = als_model.recommend(user_id=100, k=10)
    recs2 = als_model.recommend(user_id=100, k=10)
    assert recs1 == recs2
```

### Edge Case Coverage

Test boundary conditions and edge cases:

```python
def test_recommend_with_k_zero(als_model):
    """k=0 should return empty list."""
    recs = als_model.recommend(user_id=100, k=0)
    assert recs == []

def test_recommend_with_negative_k(als_model):
    """Negative k should return empty list."""
    recs = als_model.recommend(user_id=100, k=-5)
    assert recs == []

def test_recommend_with_k_larger_than_catalog(als_model):
    """k > catalog size should return all items."""
    recs = als_model.recommend(user_id=100, k=1000)
    assert len(recs) == 50  # Only 50 books in mock data
```

## Testing Strategy Comparison

### Three Levels of Testing

| Level | Speed | Isolation | Realism | Purpose |
|-------|-------|-----------|---------|---------|
| **Unit (Infrastructure)** | ⚡ Fast (5s) | 🔒 High (mock data) | 📦 Low | Test component logic |
| **Unit (Data Layer)** | 🐢 Slow (30s) | 🔓 Medium (real files) | 📊 Medium | Test artifact loading |
| **Integration** | 🐌 Slower (60s) | 🌐 Low (real API/DB) | 🎯 High | Test end-to-end flow |

**Infrastructure tests** focus on:
- ✅ Component behavior and edge cases
- ✅ API contracts and return types
- ✅ Error handling and validation
- ✅ Singleton lifecycle management

**Data layer tests** focus on:
- ✅ Artifact loading and caching
- ✅ Path resolution
- ✅ Data integrity and shape validation

**Integration tests** focus on:
- ✅ Full recommendation pipeline
- ✅ API response times and latency
- ✅ Database interactions
- ✅ Real-world scenarios

## Troubleshooting

### Tests Fail After Code Changes

1. **Clear pytest cache**:
   ```bash
   pytest --cache-clear
   ```

2. **Check for import caching**:
   ```bash
   # Restart Python interpreter or use fresh pytest run
   python -m pytest tests/unit/models/infrastructure/
   ```

### Mock Data Not Deterministic

Ensure seeding is in the fixture:
```python
@pytest.fixture
def mock_data():
    np.random.seed(42)  # ← Must be INSIDE fixture
    return generate_data()
```

### Singleton Pollution Between Tests

Always reset singletons in fixture cleanup:
```python
@pytest.fixture
def component():
    obj = Component()
    yield obj
    Component.reset()  # ← Critical
```

### FAISS Errors

If FAISS tests fail with cryptic errors:
- Check embedding dimensions match
- Verify index is not empty (check `index.ntotal > 0`)
- Ensure embeddings are float32

## Dependencies

These tests require:
- `pytest` - Test framework
- `numpy` - Numerical computing
- `pandas` - DataFrames (for metadata)
- `torch` - Deep learning framework
- `faiss-cpu` or `faiss-gpu` - Similarity search
- `unittest.mock` - Mocking (standard library)

Install via:
```bash
pip install pytest numpy pandas torch faiss-cpu
```

## Contributing

When adding new tests:

1. **Use dependency injection** - Don't load real files
2. **Seed random data** - Ensure deterministic results
3. **Test edge cases** - Empty inputs, k=0, missing entities
4. **Reset singletons** - Clean up in fixture teardown
5. **Follow naming conventions** - `test_<what>_<condition>_<expected>`

Example:
```python
def test_recommend_with_missing_user_returns_empty_list(als_model):
    """Missing user should return empty list, not error."""
    recs = als_model.recommend(user_id=999, k=10)
    assert recs == []
```

## Design Decisions

### Why Mock Data Instead of Test Fixtures?

**Mock data** (generated in tests):
- ✅ Fast - No file I/O
- ✅ Portable - Works anywhere
- ✅ Flexible - Easy to customize per test
- ✅ Clear - Test data is explicit in test file

**Test fixtures** (real but small files):
- ❌ Slower - Requires disk reads
- ❌ Environment-dependent - Needs files in place
- ❌ Brittle - Tests break if fixtures change
- ❌ Opaque - Test data hidden in files

### Why Singleton + Injectable Pattern?

**Singleton** (production):
- ✅ Performance - Cache loaded models
- ✅ Memory efficient - One copy per process

**Injectable** (testing):
- ✅ Testable - Can inject mock data
- ✅ Isolated - No shared state between tests
- ✅ Fast - No file loading

This pattern gives us the best of both worlds!

## Next Steps

After infrastructure tests pass:

1. **Run data layer tests** - Validate artifact loading
   ```bash
   pytest tests/unit/models/data/
   ```

2. **Run integration tests** - Validate end-to-end flow
   ```bash
   pytest tests/integration/models/
   ```

3. **Run full test suite** - Ensure nothing broke
   ```bash
   pytest tests/
   ```

4. **Check coverage** - Aim for 80%+ on infrastructure
   ```bash
   pytest tests/unit/models/infrastructure/ --cov=models.infrastructure
   ```
