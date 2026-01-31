# tests/unit/chatbot/tools/recsys/README.md

# Recommendation System Tools Unit Tests

This directory contains unit tests for all internal recommendation tools.

## Test Files

### Core Infrastructure
- **test_standardization.py** (16 tests) - Tests `_standardize_tool_output()` method that normalizes all retrieval tool outputs

### Retrieval Tools
- **test_semantic_search.py** (16 tests) - Semantic search with enriched metadata
- **test_als_recs.py** (13 tests) - Collaborative filtering recommendations
- **test_subject_hybrid.py** (15 tests) - Subject-based recommendations with popularity blending
- **test_popular_books.py** (17 tests) - Bayesian-ranked popular books
- **test_subject_id_search.py** (20 tests) - 3-gram TF-IDF fuzzy subject matching

### Context Tools
- **test_user_context_tools.py** (16 tests) - User profile and recent interactions tools

## Running Tests

### Run all recsys tests
```bash
pytest tests/unit/chatbot/tools/recsys/ -v
```

### Run specific test file
```bash
pytest tests/unit/chatbot/tools/recsys/test_semantic_search.py -v
```

### Run specific test class
```bash
pytest tests/unit/chatbot/tools/recsys/test_als_recs.py::TestALSRecsTool -v
```

### With coverage
```bash
pytest tests/unit/chatbot/tools/recsys/ --cov=app.agents.tools.recsys --cov-report=html
```

## Test Coverage Summary

**Total Tests: 113**

### By Component:
- Standardization: 16 tests
- Semantic Search: 16 tests
- ALS Recommendations: 13 tests
- Subject Hybrid: 15 tests
- Popular Books: 17 tests
- Subject ID Search: 20 tests
- User Context: 16 tests

### Coverage Goals:
- **Target:** >90% on `native_tools.py`
- **Excluded:** `subject_search.py` internals (TF-IDF algorithm - needs separate evaluation)

## Key Test Patterns

### Testing Retrieval Tools
All retrieval tools are tested for:
1. Correct schema (item_idx, title, author, year, num_ratings, score)
2. No enrichment metadata (except semantic_search)
3. top_k limit enforcement (1-500)
4. Error handling (returns error dict on failure)
5. Authentication/database requirements
6. Standardization integration

### Testing Semantic Search
Unique aspects:
- Meta field unpacking
- Enrichment metadata (subjects, tones, genre, vibe)
- Tone ID to name resolution
- Empty enrichment field exclusion
- Lazy searcher initialization

### Testing ALS Recommendations
Unique aspects:
- Only available for warm users (10+ ratings)
- Requires authenticated user
- Calls RecommendationService with ALS mode
- Basic metadata only

### Testing Subject Hybrid
Unique aspects:
- Requires subject indices
- Validates subject_weight (0-1)
- Calls service with subject mode
- Works for anonymous users

### Testing Popular Books
Unique aspects:
- No authentication required
- No database required (uses precomputed scores)
- Always available
- Uses Bayesian average scores
- Sorted by score descending

### Testing Subject ID Search
Unique aspects:
- JSON input/output
- Handles multiple phrases
- Fuzzy matching with TF-IDF
- Builds index on first call
- Returns candidate structure

### Testing Context Tools
Unique aspects:
- Requires allow_profile=True
- Requires authenticated user AND database
- Separate from retrieval tools
- Returns user data, not books

## Shared Fixtures

### From `conftest.py`

**mock_semantic_searcher**: Mocked SemanticSearcher
- Returns results with nested meta field
- Used by semantic_search tests

**mock_recommendation_service**: Mocked RecommendationService
- Returns RecommendedBook objects
- Used by ALS and subject_hybrid tests

**mock_load_book_meta**: Patches load_book_meta function
- Returns test DataFrame
- Used by all standardization tests

**mock_load_bayesian_scores**: Patches load_bayesian_scores
- Returns precomputed scores
- Used by popular_books tests

**mock_settings_embedder**: Patches settings.embedder
- Prevents model loading
- Used by semantic_search tests

**mock_semantic_searcher_class**: Patches SemanticSearcher class
- Prevents file system access
- Used by semantic_search tests

**mock_get_all_subject_counts**: Patches get_all_subject_counts
- Returns sample subject data
- Used by subject_id_search tests

**internal_tools_factory**: Factory for InternalTools instances
- Convenient tool creation with different configs
- Used by all tests

## Common Test Scenarios

### Test Tool Returns Correct Schema
```python
def test_returns_correct_schema(internal_tools_factory, mocks):
    tools = internal_tools_factory(db=mock_db)
    retrieval_tools = tools.get_retrieval_tools(is_warm=False)

    my_tool = next(t for t in retrieval_tools if t.name == "my_tool")
    results = my_tool.execute(...)

    assert "item_idx" in results[0]
    assert "title" in results[0]
    # ... etc
```

### Test Tool Requires Authentication
```python
def test_requires_authenticated_user(internal_tools_factory):
    tools = internal_tools_factory(current_user=None)
    retrieval_tools = tools.get_retrieval_tools(is_warm=True)

    my_tool = next(t for t in retrieval_tools if t.name == "my_tool")
    results = my_tool.execute(...)

    assert "error" in results[0]
```

### Test Tool Availability Based on Conditions
```python
def test_only_available_for_warm_users(internal_tools_factory):
    tools = internal_tools_factory(user_num_ratings=25)

    cold_tools = tools.get_retrieval_tools(is_warm=False)
    assert "my_tool" not in [t.name for t in cold_tools]

    warm_tools = tools.get_retrieval_tools(is_warm=True)
    assert "my_tool" in [t.name for t in warm_tools]
```

### Test Standardization Integration
```python
def test_adds_num_ratings(internal_tools_factory, mock_load_book_meta):
    tools = internal_tools_factory()
    retrieval_tools = tools.get_retrieval_tools(is_warm=False)

    my_tool = next(t for t in retrieval_tools if t.name == "my_tool")
    results = my_tool.execute(...)

    assert results[0]["num_ratings"] == 100  # From mock
```

## Design Principles

### 1. All Dependencies Mocked
- SemanticSearcher never loads FAISS index
- RecommendationService never loads ALS model
- No actual database queries
- No file system access

### 2. Tests Are Fast
- Each test < 0.1s
- Full suite < 5 seconds
- Heavy mocking enables speed

### 3. Tests Are Deterministic
- Same inputs always produce same outputs
- No randomness
- No external I/O

### 4. Tests Focus on Structure
- Do tools return correct schema?
- Are fields properly standardized?
- Do error cases handle gracefully?
- NOT: "Are recommendations semantically good?"

## What These Tests DON'T Cover

### Semantic Quality
- Are semantic search results actually relevant?
- Do ALS recommendations match user taste?
- Are subject matches semantically appropriate?

These require **evaluation tests** with LLM-as-judge or human evaluation (see `evaluation/chatbot/`).

### Integration Behavior
- Does SemanticSearcher actually search the FAISS index correctly?
- Does RecommendationService interact with database properly?
- Do tools work end-to-end in agent workflows?

These require **integration tests** (see `tests/integration/chatbot/tools/`).

### Algorithm Correctness
- Is the TF-IDF implementation in subject_search.py correct?
- Are Bayesian scores computed properly?
- Is the ALS model trained correctly?

These require **algorithm-specific tests** or **separate evaluation frameworks**.

## Maintenance

### When Tool Interfaces Change
1. Update fixture in `conftest.py`
2. Update affected test assertions
3. Run full test suite to catch regressions

### When Adding New Tool
1. Add mock in `conftest.py` if needed
2. Create new test file
3. Follow existing patterns for schema/error tests
4. Test tool availability conditions
5. Test standardization integration

### When Standardization Logic Changes
1. Update `test_standardization.py` first
2. Run tests to see what breaks
3. Update tool-specific tests as needed
4. Verify integration with real tools

## Debugging Tips

### Test Fails with "Tool not found"
- Check tool availability conditions (warm/cold, auth, db)
- Verify `get_retrieval_tools()` or `get_context_tools()` arguments
- Check if mocks are properly set up

### Test Fails with "Attribute not found"
- Check if mock objects have required methods
- Verify mock return values match expected structure
- Check if fixture is properly applied to test

### Test Fails Intermittently
- Look for unmocked external calls
- Check for time-dependent logic
- Verify no shared state between tests

### Standardization Test Fails
- Check `mock_load_book_meta` return value
- Verify tone map structure in `mock_db_session`
- Check if raw results have correct structure
