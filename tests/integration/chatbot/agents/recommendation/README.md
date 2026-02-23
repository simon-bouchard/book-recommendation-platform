# tests/integration/chatbot/agents/recommendation/README.md
# Recommendation Pipeline Integration Tests

Integration tests for the `RecommendationAgent` four-stage pipeline.

These tests verify **infrastructure wiring** — that data flows correctly between stages, that each stage receives the right inputs, and that fallback strategies activate when a stage fails. They make no real LLM or database calls.

For quality assessment (prose coherence, tool selection accuracy, genre filtering), see `evaluation/chatbot/recommendation_agent/`.

---

## What Is Tested

```
tests/integration/chatbot/agents/recommendation/
├── conftest.py                   # Mock builders and data factories
└── test_pipeline_integration.py  # Pipeline wiring and fallback tests
```

### `conftest.py` — Fixtures

**Mock builders** (fluent API, one per sub-agent):

| Builder | Mocks | Key methods |
|---|---|---|
| `MockPlannerBuilder` | `PlannerAgent.execute()` | `.returns_strategy()`, `.returns_warm_user_strategy()`, `.raises_error()` |
| `MockRetrievalBuilder` | `RetrievalAgent.execute()` | `.returns_batch(n)`, `.returns_candidates()`, `.returns_empty()`, `.raises_error()` |
| `MockSelectionBuilder` | `SelectionAgent.execute()` | `.returns_batch(n)`, `.returns_books()`, `.returns_empty()`, `.raises_error()` |
| `MockCurationBuilder` | `CurationAgent.execute_stream()` | `.returns_success_with_books(n)`, `.raises_error_on_stream()` |

All `execute()` methods are `AsyncMock`. `CurationAgent.execute_stream()` is a `MagicMock` whose `side_effect` returns a fresh async generator on each call — a plain `AsyncMock` cannot be used here because async generators and coroutines are distinct protocols in Python.

**Data factories:**

- `CandidateFactory` — generates raw candidate dicts (retrieval output format) and `BookRecommendation` objects (post-conversion format used by selection and curation)
- `StrategyFactory` — generates `PlannerStrategy` objects for common scenarios

### `test_pipeline_integration.py` — Test Classes

| Class | What it verifies |
|---|---|
| `TestStageTransitions` | Planner output → Retrieval input; Retrieval pool → Selection input; Selection subset → Curation input; `ExecutionContext` assembly; profile data end-to-end; metadata preservation |
| `TestParameterPropagation` | Warm/cold user ALS availability; `allow_profile` flag; `None` ratings default |
| `TestErrorHandling` | All fallback paths (see below) |
| `TestFullPipelineFlow` | Chunk sequence order; `complete` chunk contents; happy-path warm and cold users |

---

## Fallback Coverage

The orchestrator has a fallback at every stage. Each is tested independently.

| Stage failure | Expected behaviour | Tests |
|---|---|---|
| **Planning raises** | Hardcoded strategy used; warm user gets `als_recs`, cold user gets `popular_books`; pipeline continues | `test_planning_failure_pipeline_still_completes`, `test_planning_failure_warm_user_uses_als_strategy`, `test_planning_failure_cold_user_uses_popular_books_strategy` |
| **Retrieval raises** | `_retrieve_fallback_candidates()` called directly; pipeline continues | `test_retrieval_failure_invokes_direct_tool_fallback` |
| **Retrieval + fallback both raise** | Terminal error `complete` chunk emitted; selection and curation not called | `test_retrieval_total_failure_yields_error_complete_chunk` |
| **No candidates returned** | Terminal error `complete` chunk; catalog-limit hint added when query contains a year ≥ 2004 or recency keywords | `test_no_candidates_yields_terminal_error_chunk`, `test_no_candidates_with_year_query_includes_catalog_hint`, `test_no_candidates_with_recent_keyword_includes_catalog_hint` |
| **Selection raises** | Top-10 candidates from retrieval pool forwarded to curation; pipeline continues | `test_selection_failure_forwards_top10_to_curation` |
| **Selection returns empty** | Same top-10 fallback as above | `test_selection_empty_result_forwards_top10_to_curation` |
| **Curation raises** | Fallback prose token emitted (titles, authors, markdown links); `complete` chunk has `book_ids` and `success=False` | `test_curation_failure_yields_fallback_prose_token`, `test_curation_failure_complete_chunk_has_book_ids`, `test_curation_failure_complete_chunk_marks_success_false`, `test_curation_failure_fallback_prose_contains_titles` |

---

## How Tests Are Structured

All tests drive `execute_stream()` and collect chunks via the `_run()` helper:

```python
chunks, complete, tokens, statuses = await _run(agent, request)
```

`_run()` enforces the invariant that the pipeline always emits **exactly one `complete` chunk**, regardless of which stage fails. If this assertion fails, the pipeline has a control-flow bug.

Assertions then target:

```python
complete.data["success"]    # bool
complete.data["book_ids"]   # List[int]
complete.data["tools_used"] # List[str]  — annotated by orchestrator
complete.data["text"]       # str        — present on error complete chunks
```

---

## Running the Tests

```bash
# All recommendation pipeline tests
pytest tests/integration/chatbot/agents/recommendation/ -v

# Fallback tests only
pytest tests/integration/chatbot/agents/recommendation/ -v -k "Failure or fallback or no_candidates"

# Single class
pytest tests/integration/chatbot/agents/recommendation/test_pipeline_integration.py::TestErrorHandling -v
```

Tests are fully deterministic and make no external calls. Typical run time: < 2 seconds.

---

## Relation to Evaluations

| | Integration tests | Evaluations (`evaluation/chatbot/recommendation_agent/`) |
|---|---|---|
| **Purpose** | Infrastructure wiring | Output quality |
| **Sub-agents** | Mocked | Real LLMs |
| **Speed** | < 2 s | Minutes |
| **API cost** | None | Yes |
| **Deterministic** | Yes | No |
| **Runs in CI** | Yes | No |
