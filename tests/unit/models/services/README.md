# tests/unit/models/services/

Unit tests for the recommendation and similarity services layer.

## Overview

This test suite covers the Layer 5 (Services) business logic of the recommendation system:
- **RecommendationService**: Strategy selection, pipeline orchestration, and candidate enrichment
- **SimilarityService**: Subject-based, ALS-based, and hybrid book similarity with quality filtering

These tests ensure correct strategy selection based on user status, proper pipeline configuration, and accurate similarity computations across all three modes.

## Test Files

### test_recommendation_service.py

Tests for `RecommendationService` covering:
- **Strategy Selection**: Automatic selection of ALS (warm users), hybrid (cold users with preferences), or popularity (cold users without preferences)
- **Mode Forcing**: Explicit behavioral/subject mode selection via API
- **Pipeline Building**: Correct generator, fallback, filter, and ranker instantiation
- **Candidate Enrichment**: Metadata attachment and null handling
- **Logging**: Structured logging for observability
- **Edge Cases**: Empty results, invalid configurations, error handling

**Test Classes**:
- `TestRecommendationServiceInitialization`: Service initialization
- `TestStrategySelection`: Warm/cold user strategy logic
- `TestPipelineBuilding`: Pipeline component verification
- `TestCandidateEnrichment`: Metadata enrichment correctness
- `TestLogging`: Structured logging output
- `TestEdgeCases`: Boundary conditions and error cases

### test_similarity_service.py

Tests for `SimilarityService` covering:
- **Subject Mode**: Semantic similarity without filtering
- **ALS Mode**: Collaborative filtering with 10+ rating threshold
- **Hybrid Mode**: Blended scores with 5+ rating threshold and configurable alpha
- **Two-Pool Architecture**: Query any book, results from high-quality candidates only
- **Filtering**: Configurable min_rating_count and filter_candidates parameters
- **Result Formatting**: Metadata enrichment for display
- **Logging**: Structured logging for similarity searches

**Test Classes**:
- `TestSimilarityServiceInitialization`: Service initialization
- `TestSubjectSimilarity`: Subject-based similarity search
- `TestALSSimilarity`: Collaborative filtering similarity with filtering
- `TestHybridSimilarity`: Blended similarity with complex score computation
- `TestTwoPoolArchitecture`: Query/candidate pool separation
- `TestFiltering`: Quality threshold enforcement
- `TestResultFormatting`: Metadata attachment
- `TestLogging`: Structured logging output
- `TestEdgeCases`: Boundary conditions and error cases

## Running Tests

### Run All Service Tests
```bash
pytest tests/unit/models/services/ -v
```

### Run Specific Test File
```bash
pytest tests/unit/models/services/test_recommendation_service.py -v
pytest tests/unit/models/services/test_similarity_service.py -v
```

### Run Specific Test Class
```bash
pytest tests/unit/models/services/test_recommendation_service.py::TestStrategySelection -v
pytest tests/unit/models/services/test_similarity_service.py::TestHybridSimilarity -v
```

### Run Specific Test
```bash
pytest tests/unit/models/services/test_recommendation_service.py::TestStrategySelection::test_warm_user_gets_als_pipeline -v
```

### Run with Coverage
```bash
pytest tests/unit/models/services/ --cov=models.services --cov-report=html
```

### Run in Parallel (faster)
```bash
pytest tests/unit/models/services/ -n auto
```

## Key Fixtures

### Shared Fixtures (both test files)

#### `mock_book_meta`
- **Returns**: DataFrame with book metadata (title, author, year, ratings, etc.)
- **Index**: item_idx values
- **Use**: Testing metadata enrichment and filtering

#### User Fixtures (test_recommendation_service.py)

##### `warm_user`
- **Returns**: User with user_id=123, has ALS factors
- **Use**: Testing warm user strategy (ALS-based recommendations)

##### `cold_user_with_prefs`
- **Returns**: User with user_id=456, has subject preferences but no ALS factors
- **Use**: Testing cold user with preferences strategy (hybrid recommendations)

##### `cold_user_no_prefs`
- **Returns**: User with user_id=789, fav_subjects=[PAD_IDX]
- **Use**: Testing cold user without preferences strategy (popularity fallback)

#### Embedding Fixtures (test_similarity_service.py)

##### `mock_subject_embeddings`
- **Returns**: (embeddings, ids) - normalized random embeddings for 100 books
- **Use**: Testing subject similarity computations

##### `mock_als_factors`
- **Returns**: (factors, row_to_id_map) - random ALS factors for 80 books
- **Use**: Testing ALS similarity computations

##### `mock_db`
- **Returns**: Mock database session
- **Use**: Testing database-dependent operations (filtering read books)

## Testing Patterns

### 1. Strategy Selection Testing

Tests verify correct generator selection based on user status:

```python
def test_warm_user_gets_als_pipeline(self, warm_user, mock_db):
    """Warm user should use ALS-based pipeline."""
    # Mock ALSModel.has_user() to return True
    # Verify ALSBasedGenerator is selected
    # Verify BayesianPopularityGenerator is used as fallback
```

**Key Pattern**: Mock `ALSModel.has_user()` to control warm/cold user status.

### 2. Pipeline Component Testing

Tests verify correct pipeline construction:

```python
def test_pipeline_has_correct_components(self, warm_user):
    """Pipeline should have generator, fallback, filter, and ranker."""
    # Mock pipeline class to intercept construction
    # Verify all components are correct types
    # Verify ReadBooksFilter is used
```

**Key Pattern**: Mock `RecommendationPipeline` class to inspect constructor arguments.

### 3. Similarity Mode Testing

Tests verify correct similarity computations for each mode:

```python
def test_subject_similarity_no_filtering(self):
    """Subject mode should not filter low-rated books."""
    # Mock subject embeddings
    # Call service.get_similar(mode="subject")
    # Verify all books can appear in results (no filtering)
```

**Key Pattern**: Mock `load_book_subject_embeddings()` and verify index construction.

### 4. Filtering Logic Testing

Tests verify quality thresholds are enforced:

```python
def test_als_filters_low_rated_books(self):
    """ALS mode should only return books with 10+ ratings."""
    # Mock ALS factors and book metadata
    # Call service.get_similar(mode="als")
    # Verify all results have >= 10 ratings
```

**Key Pattern**: Mock `load_book_meta()` with varying rating counts.

### 5. Hybrid Blending Testing

Tests verify correct score blending:

```python
def test_hybrid_blends_scores_correctly(self):
    """Hybrid mode should blend subject and ALS scores with alpha weight."""
    # Mock both subject embeddings and ALS factors
    # Call with alpha=0.7
    # Compute expected blended scores manually
    # Verify results match expected ordering
```

**Key Pattern**: Mock all data sources and manually compute expected scores.

### 6. Logging Testing

Tests verify structured logging output:

```python
def test_logs_recommendation_started(self, warm_user, mock_db, caplog):
    """Should log recommendation start with user context."""
    with caplog.at_level(logging.INFO):
        service.recommend(warm_user, config, mock_db)

    assert "Recommendation started" in caplog.text
    assert "user_id=123" in caplog.text
```

**Key Pattern**: Use pytest's `caplog` fixture to capture and verify log output.

## Common Testing Gotchas

### 1. Always Use PAD_IDX Constant

**Wrong**:
```python
user = User(user_id=789, fav_subjects=[0])  # Hardcoded!
```

**Right**:
```python
from models.core.constants import PAD_IDX
user = User(user_id=789, fav_subjects=[PAD_IDX])
```

**Why**: PAD_IDX can be configured via environment variable. Hardcoding breaks tests.

### 2. Properly Connect Mock Return Values

**Wrong**:
```python
mock_pipeline = Mock()
mock_pipeline_class.return_value = ...  # Oops, forgot this line!
```

**Right**:
```python
mock_pipeline = Mock()
mock_pipeline.recommend.return_value = []
mock_pipeline_class.return_value = mock_pipeline  # Connect instance to class
```

**Why**: Without connecting, service gets a different Mock instance.

### 3. Mock at the Right Import Path

**Wrong**:
```python
@patch("models.infrastructure.als_model.ALSModel")  # Too deep!
```

**Right**:
```python
@patch("models.services.recommendation_service.ALSModel")  # Where it's imported
```

**Why**: Mock must patch where the name is looked up, not where it's defined.

### 4. Verify Mock Calls with call_args

**Wrong**:
```python
assert mock_pipeline.recommend.called  # Only checks if called
```

**Right**:
```python
call_args = mock_pipeline.recommend.call_args
assert call_args[0][1] == 20  # Verify k parameter
assert call_args[0][2] == mock_db  # Verify db parameter
```

**Why**: Verifying parameters ensures correct values are passed.

## Adding New Tests

### 1. Test a New Strategy Selection Rule

```python
def test_new_user_type_gets_special_pipeline(self, new_user_type, mock_db):
    """New user type should use custom pipeline."""
    # 1. Create fixture for new user type
    # 2. Mock necessary infrastructure (ALSModel, etc.)
    # 3. Mock RecommendationPipeline to intercept construction
    # 4. Call service.recommend()
    # 5. Verify correct generator was selected

    with patch("models.services.recommendation_service.RecommendationPipeline") as mock_pipeline_class:
        mock_pipeline = Mock()
        mock_pipeline.recommend.return_value = []
        mock_pipeline_class.return_value = mock_pipeline

        service = RecommendationService()
        config = RecommendationConfig.default()

        service.recommend(new_user_type, config, mock_db)

        call_args = mock_pipeline_class.call_args
        generator = call_args[1]["generator"]
        assert generator.__class__.__name__ == "ExpectedGeneratorType"
```

### 2. Test a New Similarity Mode

```python
def test_new_similarity_mode(self):
    """New similarity mode should compute correct scores."""
    # 1. Mock required data sources (embeddings, factors, etc.)
    # 2. Call service.get_similar(mode="new_mode")
    # 3. Manually compute expected scores
    # 4. Verify results match expectations

    with patch("models.services.similarity_service.load_new_embeddings") as mock_load:
        mock_load.return_value = (mock_embeddings, mock_ids)

        service = SimilarityService()
        results = service.get_similar(item_idx=1050, mode="new_mode", k=10)

        # Verify results
        assert len(results) == 10
        assert results[0]["score"] > results[1]["score"]  # Sorted descending
```

### 3. Test a New Filtering Rule

```python
def test_new_filter_criterion(self):
    """New filter should enforce quality criterion."""
    # 1. Mock book_meta with varying quality metrics
    # 2. Call similarity service with new filter enabled
    # 3. Verify all results meet criterion

    with patch("models.services.similarity_service.load_book_meta") as mock_meta:
        # Books with varying quality scores
        mock_meta.return_value = pd.DataFrame({
            "quality_score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }, index=range(1000, 1010))

        service = SimilarityService()
        results = service.get_similar(item_idx=1000, min_quality=5)

        # All results should have quality >= 5
        for result in results:
            assert mock_meta.return_value.loc[result["item_idx"], "quality_score"] >= 5
```

## Test Coverage Goals

- **Line Coverage**: > 90%
- **Branch Coverage**: > 85%
- **All Business Rules**: 100% coverage
- **All Edge Cases**: Comprehensive coverage

### Current Coverage

Run to check:
```bash
pytest tests/unit/models/services/ --cov=models.services --cov-report=term-missing
```

## Debugging Failed Tests

### 1. Use -vv for Verbose Output
```bash
pytest tests/unit/models/services/test_recommendation_service.py::failing_test -vv
```

### 2. Use --pdb to Drop into Debugger
```bash
pytest tests/unit/models/services/test_recommendation_service.py::failing_test --pdb
```

### 3. Print Mock Call History
```python
def test_debugging(self, mock_db):
    service.recommend(user, config, mock_db)

    # Print all calls to mock
    print(mock_pipeline.recommend.call_args_list)
    print(mock_pipeline.recommend.call_count)
```

### 4. Verify Mock Setup
```python
def test_debugging(self):
    # Check mock was set up correctly
    assert mock_pipeline_class.return_value is mock_pipeline
    assert mock_pipeline.recommend.return_value == []
```

## Dependencies

These tests require:
- `pytest >= 7.0`
- `pytest-mock` (for monkeypatch fixture)
- `pandas` (for DataFrame mocking)
- `numpy` (for embedding mocking)

Install with:
```bash
pip install pytest pytest-mock pandas numpy
```

## Related Documentation

- **Service Implementation**: `models/services/recommendation_service.py`
- **Domain Layer Tests**: `tests/unit/models/domain/`
- **Infrastructure Layer Tests**: `tests/unit/models/infrastructure/`
- **Integration Tests**: `tests/integration/models/`
- **Refactor Plan**: `models/models_refactor_plan.md`

## Best Practices

1. **Test Business Logic, Not Implementation Details**
   - Test what the service does, not how it does it
   - Mock infrastructure, not domain logic

2. **Use Descriptive Test Names**
   - `test_warm_user_gets_als_pipeline` > `test_pipeline_1`

3. **One Assertion per Test (when possible)**
   - Makes failures easier to diagnose

4. **Arrange-Act-Assert Pattern**
   ```python
   # Arrange: Set up mocks and fixtures
   mock_pipeline = Mock()

   # Act: Call the service
   result = service.recommend(user, config, db)

   # Assert: Verify expectations
   assert result is not None
   ```

5. **Keep Tests Independent**
   - Each test should be able to run in isolation
   - Don't rely on test execution order

6. **Mock External Dependencies**
   - Always mock database, file I/O, network calls
   - Never mock the system under test

## Troubleshooting

### Tests Pass Locally But Fail in CI

**Possible Causes**:
- Environment variable differences (check PAD_IDX)
- Different numpy/pandas versions
- Race conditions in parallel test runs

**Solutions**:
```bash
# Run tests with same environment as CI
docker run --rm -v $(pwd):/app python:3.10 pytest tests/unit/models/services/

# Check environment variables
env | grep PAD_IDX
```

### Mock Not Working As Expected

**Check**:
1. Are you mocking at the right import path?
2. Did you set `return_value` for the mock?
3. Is the mock being created before the service is instantiated?

**Debug**:
```python
# Add to test
print(f"Mock class: {mock_pipeline_class}")
print(f"Mock instance: {mock_pipeline}")
print(f"Return value: {mock_pipeline_class.return_value}")
print(f"Called: {mock_pipeline.recommend.called}")
```

### Assertion Failures on Mock Calls

**Common Issue**: Checking call_args on wrong mock instance

**Solution**: Verify you're checking the right mock:
```python
# Wrong: Checking class when instance was called
assert mock_pipeline_class.recommend.called

# Right: Checking instance
assert mock_pipeline.recommend.called
```

## Contributing

When adding new tests:
1. Follow existing test structure (class organization)
2. Add docstrings explaining what is tested
3. Use descriptive variable names
4. Include both happy path and edge cases
5. Update this README if adding new fixtures or patterns

## Questions?

For questions about these tests, see:
- Implementation: `models/services/`
- Architecture: `models/models_refactor_plan.md`
- Team contact: Your team lead or architecture owner
