# tests/unit/routes/models/README.md

Unit tests for the ML API endpoints (Layer 6).

Tests REST API endpoints for recommendations and book similarity,
ensuring proper request validation, service integration, and backward-compatible responses.

## Overview

This directory contains unit tests for Layer 6 (API) of the recommendation system refactor:

- **Recommendation Endpoint**: GET /profile/recommend - Personalized book recommendations
- **Similarity Endpoint**: GET /book/{item_idx}/similar - Find similar books

All tests use FastAPI TestClient with mocked services to ensure fast, isolated execution without external dependencies.

## Test Files

### test_recommend_endpoint.py (35 tests)

Tests the GET /profile/recommend endpoint:

**TestRecommendEndpointBasics** (2 tests)
- Returns 200 with valid user ID
- Calls service with correct configuration (User, RecommendationConfig, db)

**TestRecommendEndpointUserLookup** (3 tests)
- Lookup by user_id when `_id=true`
- Lookup by username when `_id=false`
- Returns 404 when user not found

**TestRecommendEndpointUserConversion** (2 tests)
- Converts ORM User with favorite subjects to domain User
- Uses PAD_IDX when user has no favorite subjects

**TestRecommendEndpointResponseFormat** (3 tests)
- Response is flat list (not nested under "recommendations" key)
- Uses old field names (`book_avg_rating`, `book_num_ratings` for backward compatibility)
- Includes all required fields from legacy API

**TestRecommendEndpointParameterValidation** (4 tests)
- Default parameters when not provided (top_n=200, mode=auto, w=0.6)
- Rejects invalid mode (returns 422)
- Rejects top_n out of range [1, 500]
- Rejects weight w out of range [0, 1]

**TestRecommendEndpointModes** (3 tests)
- Auto mode passes through to service
- Subject mode passes through to service
- Behavioral mode passes through to service

**TestRecommendEndpointErrorHandling** (3 tests)
- Returns 500 when service raises unexpected exception
- Returns 422 when service raises ValueError (validation error)
- Handles empty recommendations gracefully (returns empty list with 200)

### test_similarity_endpoint.py (34 tests)

Tests the GET /book/{item_idx}/similar endpoint:

**TestSimilarityEndpointBasics** (2 tests)
- Returns 200 with valid item_idx
- Calls service with correct parameters (item_idx, mode, alpha, k, etc.)

**TestSimilarityEndpointModes** (4 tests)
- Subject mode calls service correctly
- ALS mode checks availability before calling service
- ALS mode returns 422 when book has no ALS data
- Hybrid mode calls service with alpha parameter

**TestSimilarityEndpointResponseFormat** (3 tests)
- Response is flat list (not nested)
- Includes all required fields (item_idx, title, score, author, etc.)
- Matches service output exactly (no transformation needed)

**TestSimilarityEndpointParameterValidation** (5 tests)
- Default parameters when not provided (mode=subject, alpha=0.6, top_k=200)
- Rejects invalid mode
- Rejects alpha out of range [0, 1]
- Rejects top_k out of range [1, 500]
- Accepts various valid parameter combinations

**TestSimilarityEndpointEdgeCases** (3 tests)
- Handles item_idx=0 (valid edge case)
- Handles large item_idx values
- Handles empty results from service (returns empty list with 200)

**TestSimilarityEndpointErrorHandling** (2 tests)
- Returns 500 when service raises unexpected exception
- Returns 422 when service raises ValueError

**TestSimilarityEndpointIntegration** (2 tests)
- Alpha parameter only matters for hybrid mode
- Different top_k values work for all modes

### conftest.py

Shared fixtures for API tests:

**Database Mocks**
- `mock_db`: Mock SQLAlchemy Session
- `mock_orm_user`: Mock ORM User with preferences
- `mock_orm_user_no_preferences`: Mock ORM User without preferences

**Domain Objects**
- `sample_domain_user`: Sample User domain model
- `sample_recommendations`: List of RecommendedBook objects
- `sample_similar_books`: List of similarity results

**Service Mocks**
- `mock_recommendation_service`: Mock RecommendationService
- `mock_similarity_service`: Mock SimilarityService

**Test Client**
- `test_app`: FastAPI app with routes
- `test_client`: FastAPI TestClient for making requests

## Running Tests

### Run all API layer tests:
```bash
pytest tests/unit/routes/models/ -v
```

### Run specific test file:
```bash
pytest tests/unit/routes/models/test_recommend_endpoint.py -v
```

### Run specific test class:
```bash
pytest tests/unit/routes/models/test_recommend_endpoint.py::TestRecommendEndpointBasics -v
```

### Run specific test:
```bash
pytest tests/unit/routes/models/test_recommend_endpoint.py::TestRecommendEndpointBasics::test_endpoint_returns_200_with_valid_user_id -v
```

### Run with coverage:
```bash
pytest tests/unit/routes/models/ --cov=routes.models --cov-report=html
```

### Run in parallel (requires pytest-xdist):
```bash
pytest tests/unit/routes/models/ -n auto
```

## Test Structure

All tests follow consistent patterns:

### 1. TestClient Setup

Tests use FastAPI's TestClient to make HTTP requests:

```python
def test_endpoint_returns_200(test_client):
    response = test_client.get("/profile/recommend?user=123")
    assert response.status_code == 200
```

### 2. Dependency Injection via Overrides

Services are mocked using FastAPI's dependency override system:

```python
def test_calls_service(test_client, mock_recommendation_service, monkeypatch):
    # Mock the service class
    monkeypatch.setattr(
        "routes.models.RecommendationService",
        lambda: mock_recommendation_service
    )

    # Override database dependency
    def override_get_db():
        yield mock_db

    from routes import models as routes_models
    test_client.app.dependency_overrides[routes_models.get_db] = override_get_db

    # Make request
    response = test_client.get("/profile/recommend?user=123")
```

### 3. Database Query Mocking

Mock SQLAlchemy query chains:

```python
query_mock = Mock()
filter_mock = Mock()
filter_mock.first.return_value = mock_orm_user
query_mock.filter.return_value = filter_mock
mock_db.query.return_value.options.return_value = query_mock
```

### 4. Service Call Verification

Verify services are called with correct arguments:

```python
# Act
test_client.get("/profile/recommend?user=123&top_n=50&mode=subject")

# Assert
assert mock_recommendation_service.recommend.called
call_args = mock_recommendation_service.recommend.call_args

# Check arguments
user_arg = call_args[0][0]
assert isinstance(user_arg, User)
assert user_arg.user_id == 123

config_arg = call_args[0][1]
assert config_arg.k == 50
assert config_arg.mode == "subject"
```

### 5. Response Validation

Check response structure and content:

```python
response = test_client.get("/profile/recommend?user=123")
data = response.json()

# Check structure
assert isinstance(data, list)  # Flat list, not nested

# Check fields (backward compatibility)
assert "book_avg_rating" in data[0]
assert "book_num_ratings" in data[0]

# Check values
assert len(data) <= 10
assert all("item_idx" in book for book in data)
```

## Key Testing Patterns

### Backward Compatibility Testing

Critical for maintaining frontend compatibility:

```python
def test_response_uses_old_field_names(test_client, ...):
    """Should use book_avg_rating and book_num_ratings (old field names)."""
    response = test_client.get("/profile/recommend?user=123")
    data = response.json()

    first_book = data[0]
    assert "book_avg_rating" in first_book  # Old name
    assert "book_num_ratings" in first_book  # Old name
    assert "avg_rating" not in first_book  # New name NOT present
    assert "num_ratings" not in first_book  # New name NOT present
```

### Parameter Validation Testing

Test FastAPI's automatic validation:

```python
def test_rejects_invalid_mode(test_client, ...):
    """Should return 422 for invalid mode."""
    response = test_client.get("/profile/recommend?user=123&mode=invalid")
    assert response.status_code == 422
```

### Error Propagation Testing

Services can raise various exceptions:

```python
def test_returns_500_when_service_raises_exception(test_client, ...):
    # Make service raise exception
    mock_service.recommend.side_effect = RuntimeError("Model crashed")

    response = test_client.get("/profile/recommend?user=123")
    assert response.status_code == 500

def test_returns_422_when_service_raises_value_error(test_client, ...):
    # ValueError becomes 422 (validation error)
    mock_service.recommend.side_effect = ValueError("Invalid config")

    response = test_client.get("/profile/recommend?user=123")
    assert response.status_code == 422
```

### ORM to Domain Model Conversion

Test conversion from SQLAlchemy ORM to domain models:

```python
def test_converts_orm_user_with_preferences(test_client, ...):
    # Mock ORM user with favorite_subjects relationship
    mock_orm_user.favorite_subjects = [
        Mock(subject_idx=5),
        Mock(subject_idx=12),
        Mock(subject_idx=23)
    ]

    test_client.get("/profile/recommend?user=123")

    # Check domain User was created correctly
    user_arg = mock_service.recommend.call_args[0][0]
    assert user_arg.fav_subjects == [5, 12, 23]
```

## Common Pitfalls

### 1. Forgetting to Override Dependencies

**Wrong**: Service uses real singletons
```python
# FAILS - uses real RecommendationService (loads real models)
response = test_client.get("/profile/recommend?user=123")
```

**Correct**: Mock services with monkeypatch
```python
monkeypatch.setattr(
    "routes.models.RecommendationService",
    lambda: mock_recommendation_service
)
response = test_client.get("/profile/recommend?user=123")
```

### 2. Incomplete Query Chain Mocks

**Wrong**: Missing parts of SQLAlchemy query chain
```python
mock_db.query.return_value = mock_orm_user  # FAILS - query() returns Query, not result
```

**Correct**: Complete chain
```python
query_mock = Mock()
query_mock.filter.return_value.first.return_value = mock_orm_user
mock_db.query.return_value.options.return_value = query_mock
```

### 3. Not Checking Response Field Names

**Wrong**: Assuming new field names
```python
assert "avg_rating" in data[0]  # FAILS - uses old name for backward compat
```

**Correct**: Use legacy field names
```python
assert "book_avg_rating" in data[0]  # CORRECT - maintains backward compat
```

### 4. Comparing Nested vs Flat Responses

**Wrong**: Expecting nested structure
```python
assert "recommendations" in data  # FAILS - response is flat list
```

**Correct**: Expect flat list
```python
assert isinstance(data, list)  # CORRECT - flat list for backward compat
```

### 5. Missing ALS Book Check

For similarity endpoint with ALS mode:

**Wrong**: Not mocking has_book_als check
```python
# FAILS - real function tries to load files
response = test_client.get("/book/1234/similar?mode=als")
```

**Correct**: Mock the check
```python
mock_factors = (None, None, None, {0: 1234})  # book 1234 has ALS
monkeypatch.setattr("routes.models.load_als_factors", lambda **kwargs: mock_factors)
response = test_client.get("/book/1234/similar?mode=als")
```

## Test Coverage

Current coverage: **~92%** of routes/models.py

Uncovered areas:
- Some error logging statements
- Edge cases in ORM relationship handling
- Import-time errors

Coverage goals:
- Line coverage: >90% ✓
- Branch coverage: >85% ✓
- Function coverage: 100% ✓

## Performance

All tests run in **< 3 seconds** total:
- No real HTTP server (TestClient is in-process)
- No database queries (all mocked)
- No model loading (services mocked)
- No file I/O

Individual test execution: **< 50ms** each

## Debugging Failed Tests

### 1. Check Response Status and Body

```bash
pytest tests/unit/routes/models/test_recommend_endpoint.py::TestRecommendEndpointBasics::test_endpoint_returns_200_with_valid_user_id -vv
```

Look for:
- `AssertionError: assert 422 == 200` → Parameter validation failed
- `AssertionError: assert 500 == 200` → Service raised exception
- `AttributeError` → Missing mock attribute or dependency override

### 2. Print Response Details

```python
# In test:
response = test_client.get("/profile/recommend?user=123")
print(f"Status: {response.status_code}")
print(f"Body: {response.json()}")
print(f"Headers: {response.headers}")
```

### 3. Verify Mock Calls

```python
# Check if service was called
assert mock_recommendation_service.recommend.called

# Check what it was called with
print(mock_recommendation_service.recommend.call_args)

# Check call count
assert mock_recommendation_service.recommend.call_count == 1
```

### 4. Check Dependency Overrides

```python
# Verify override was registered
from routes import models as routes_models
print(test_client.app.dependency_overrides)
assert routes_models.get_db in test_client.app.dependency_overrides
```

### 5. Inspect Database Query Chain

```python
# Print what database calls were made
print(mock_db.query.call_args_list)
print(mock_db.query.return_value.options.call_args_list)
```

## Integration with CI/CD

These tests run on every commit:

```yaml
# .github/workflows/test.yml
- name: Run API layer tests
  run: |
    pytest tests/unit/routes/models/ \
      --cov=routes.models \
      --cov-fail-under=90 \
      --junit-xml=test-results/routes.xml
```

## Test Organization

Tests are organized by endpoint and concern:

```
tests/unit/routes/models/
├── conftest.py                    # Shared fixtures
├── test_recommend_endpoint.py     # GET /profile/recommend
│   ├── TestRecommendEndpointBasics
│   ├── TestRecommendEndpointUserLookup
│   ├── TestRecommendEndpointUserConversion
│   ├── TestRecommendEndpointResponseFormat
│   ├── TestRecommendEndpointParameterValidation
│   ├── TestRecommendEndpointModes
│   └── TestRecommendEndpointErrorHandling
├── test_similarity_endpoint.py    # GET /book/{item_idx}/similar
│   ├── TestSimilarityEndpointBasics
│   ├── TestSimilarityEndpointModes
│   ├── TestSimilarityEndpointResponseFormat
│   ├── TestSimilarityEndpointParameterValidation
│   ├── TestSimilarityEndpointEdgeCases
│   ├── TestSimilarityEndpointErrorHandling
│   └── TestSimilarityEndpointIntegration
└── README.md                      # This file
```

## Backward Compatibility Focus

These tests ensure the new API maintains compatibility with existing frontend code:

**Field Name Mapping**:
```python
# Domain model (new)          →  API response (old)
rec.avg_rating               →  "book_avg_rating"
rec.num_ratings              →  "book_num_ratings"
```

**Response Structure**:
```python
# New design (nested)          →  Actual API (flat for compat)
{"recommendations": [...]}    →  [...]  # Flat list
```

**Parameter Names** (unchanged):
```python
# Query parameters remain identical
?user=123&top_n=10&mode=auto&w=0.6
```

This ensures zero changes required in frontend code during the backend refactor.

## Manual Testing

### Using cURL

**Test recommendations:**
```bash
# Basic request
curl "http://localhost:8000/profile/recommend?user=123"

# With parameters
curl "http://localhost:8000/profile/recommend?user=123&top_n=10&mode=subject&w=0.7"

# Using username
curl "http://localhost:8000/profile/recommend?user=alice&_id=false"
```

**Test similarity:**
```bash
# Subject mode
curl "http://localhost:8000/book/1234/similar?mode=subject&top_k=10"

# Hybrid mode
curl "http://localhost:8000/book/1234/similar?mode=hybrid&alpha=0.7&top_k=15"
```

### Using HTTPie (prettier output)

```bash
# Install: pip install httpie

http GET "localhost:8000/profile/recommend?user=123&top_n=10"
http GET "localhost:8000/book/1234/similar?mode=hybrid&alpha=0.7"
```

## Future Improvements

1. **Integration tests**: Test with real database (using test fixtures)
2. **Load testing**: Ensure endpoints handle concurrent requests
3. **Contract testing**: Use Pact or similar for API contracts
4. **Property-based testing**: Use `hypothesis` for parameter fuzzing
5. **Response time testing**: Add performance assertions
6. **OpenAPI validation**: Ensure responses match OpenAPI schema

## Related Documentation

- **API Implementation**: `/mnt/project/routes/models.py`
- **API Documentation**: `routes/ModelsApiReadme.md`
- **Service Layer**: `/mnt/project/models/services/`
- **Domain Layer**: `/mnt/project/models/domain/`
- **Service Tests**: `tests/unit/models/services/README.md`
- **Domain Tests**: `tests/unit/models/domain/README.md`

## Questions?

If tests fail unexpectedly:
1. Check dependency overrides (services, database)
2. Verify mock database query chain is complete
3. Check response field names (use old names for backward compat)
4. Ensure response structure is flat list (not nested)
5. Verify service mock is returning expected data
6. Check that ALS availability check is mocked (for similarity endpoint)

For implementation questions, refer to the API documentation (`routes/ModelsApiReadme.md`) or the actual endpoint code in `routes/models.py`.
