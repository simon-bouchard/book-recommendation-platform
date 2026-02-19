# tests/unit/chatbot/tools/recsys/README.md

# Recommendation System Tools Unit Tests

Unit tests for `InternalTools` in `app/agents/tools/recsys/native_tools.py`.

## Test Files

- **test_tool_gating.py** (11 tests) — `get_retrieval_tools()` and `get_context_tools()` conditional inclusion logic
- **test_standardization.py** (11 tests) — `_standardize_tool_output()` field normalization logic

## What These Tests Cover

### Tool Gating (`test_tool_gating.py`)
Verifies which tools are included or excluded based on runtime conditions:
- The three always-present tools (`book_semantic_search`, `subject_id_search`, `popular_books`) appear regardless of user, db, or warm state
- `subject_hybrid_pool` requires a database connection
- `als_recs` requires all three: `is_warm=True`, authenticated user, and database
- Context tools (`user_profile`, `recent_interactions`) require `allow_profile=True`, authenticated user, and database

All `_create_*_tool()` methods are replaced with lightweight stubs — no database, FAISS index, or any other I/O involved.

### Standardization (`test_standardization.py`)
Verifies `_standardize_tool_output()` produces the correct schema:
- Empty input returns empty list
- Dicts with `error` key pass through unchanged
- Books missing `item_idx` are silently dropped
- `num_ratings` is populated from `book_meta` for known books, defaults to `0` for unknown
- `cover_id` is populated from `book_meta` when present and truthy, `None` otherwise
- Output contains exactly the expected core fields — no extras stripped or added
- `score` is preserved from input unchanged

`load_book_meta()` is the only dependency, patched inline in each test.

## What These Tests Don't Cover

### Tool behavior
Whether tools return correct results, handle errors, enforce `top_k` bounds, or call
services correctly. This is covered by integration tests at
`tests/integration/chatbot/tools/test_retrieval_tools.py`.

### Semantic quality
Whether recommendations are actually relevant or appropriate.
This requires evaluation tests with LLM-as-judge (see `evaluation/chatbot/`).

## Running Tests

```bash
# Run all recsys unit tests
pytest tests/unit/chatbot/tools/recsys/ -v

# Run specific file
pytest tests/unit/chatbot/tools/recsys/test_tool_gating.py -v
pytest tests/unit/chatbot/tools/recsys/test_standardization.py -v

# With coverage
pytest tests/unit/chatbot/tools/recsys/ --cov=app.agents.tools.recsys --cov-report=html
```

## Design Principles

**No I/O** — all external dependencies are mocked. Tests complete in milliseconds.

**Minimal mocking surface** — gating tests stub `_create_*_tool()` at the instance level rather than patching deep dependencies. Standardization tests patch only `load_book_meta`.

**Pin the contract** — `test_output_contains_exactly_core_fields` explicitly asserts the full set of output keys so any accidental addition or removal of fields is caught immediately.
