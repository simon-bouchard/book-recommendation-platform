# tests/unit/recsys/README.md
# Recsys Unit Tests

Unit tests for pure methods in the recommendation pipeline agents.

These tests have no external dependencies — no LLM calls, no database, no filesystem access. All agent constructors are patched at the fixture level. Typical run time: < 3 seconds for all 90 tests.

---

## Structure

```
tests/unit/recsys/
├── conftest.py                  # Patched agent fixtures and data factories
├── test_curation_logic.py       # CurationAgent pure methods
├── test_planner_parsing.py      # PlannerAgent pure methods
└── test_orchestrator_init.py    # Orchestrator init logic and pure helpers
```

---

## What Is Tested

### `test_curation_logic.py` — `CurationAgent`

| Method | What is verified |
|---|---|
| `_extract_book_ids_from_citations(text)` | Single and multiple citations; citation order preserved; duplicates deduplicated at first occurrence; URL-style links ignored; non-integer targets ignored; empty text |
| `_order_books_by_citations(candidates, ids)` | Reorders to match citation order; uncited books dropped; cited ID not in candidates silently skipped; empty inputs |
| `_prepare_candidates(candidates)` | Uses `to_curation_dict()` when available; manual extraction fallback; vibe truncated at 200 chars with `"..."`; enrichment fields absent when `None`; order preserved |

### `test_planner_parsing.py` — `PlannerAgent`

| Method | What is verified |
|---|---|
| `_parse_strategy_response(text, profile_data)` | Plain JSON; ` ```json ` fence stripped; plain ` ``` ` fence stripped; profile data attached from argument (not LLM output); `negative_constraints` parsed when present, `None` when absent; `KeyError` on missing required fields; `json.JSONDecodeError` on invalid input |
| `_build_prompt(query, tools, profile_data)` | Query present; tools listed; user rating count and ALS flag included; `"None (no profile available)"` written when `profile_data=None`; subjects and IDs appear when profile provided; recent interactions appear when provided |

### `test_orchestrator_init.py` — `RecommendationAgent`

| Target | What is verified |
|---|---|
| `__init__` | `_has_als_recs` set correctly for warm/cold/boundary/custom threshold; `None` ratings default to 0; `allow_profile` stored; injected sub-agents used as-is |
| `_build_fallback_strategy()` | Warm user → `als_recs` primary, `popular_books` fallback; cold user → `popular_books` primary; profile data always `None`; returns `PlannerStrategy` |
| `_build_curation_fallback_text(candidates)` | Header present; all titles and authors included; `[Title](item_idx)` citation format used; handles empty list and `None` author |
| `_no_candidates_complete_chunk(query, start_time)` | `success=False`; `book_ids=[]`; catalog-limit hint (mentioning 2004) triggered by years 2004–2025 or keywords `recent`/`new`/`latest`; not triggered by years before 2004 or generic queries |

---

## Fixture Design

Constructor dependencies are patched in `conftest.py` so agents can be instantiated without triggering LLM initialisation, prompt file I/O, or logging setup:

```python
# conftest.py pattern
with (
    patch("...read_prompt", return_value="prompt"),
    patch("...get_llm", return_value=MagicMock()),
    patch("...create_react_agent", return_value=MagicMock()),
    patch("...append_chatbot_log"),
):
    return CurationAgent()
```

`test_orchestrator_init.py` uses a module-level `_make_orchestrator(num_ratings, **kwargs)` helper rather than a fixture because many tests need to vary `num_ratings` and `warm_threshold` together inline.

---

## Running the Tests

```bash
# All recsys unit tests
pytest tests/unit/recsys/ -v

# Single file
pytest tests/unit/recsys/test_curation_logic.py -v

# Single class
pytest tests/unit/recsys/test_orchestrator_init.py::TestBuildFallbackStrategy -v
```

---

## Relation to Integration Tests and Evaluations

| | Unit tests (here) | Integration tests | Evaluations |
|---|---|---|---|
| **Tests** | Pure logic, edge cases | Stage wiring, fallback triggers | Output quality |
| **Sub-agents** | Real class, patched constructor | All four mocked | Real LLMs |
| **Speed** | < 3 s | < 5 s | Minutes |
| **API cost** | None | None | Yes |
| **Runs in CI** | Yes | Yes | No |
