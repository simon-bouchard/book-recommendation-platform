# Router Evaluation

Tests RouterLLM classification accuracy with real LLM queries.

## Structure

```
evaluation/chatbot/router/
├── README.md                          # This file
├── evaluate_router.py                 # Stateless evaluation
├── router_test_queries.json           # Stateless test cases
├── evaluate_router_with_history.py    # History/context evaluation
├── router_conversation_tests.json     # Conversation test cases
└── results/                           # Output directory (auto-created)
    ├── router_eval_TIMESTAMP.json
    └── router_history_eval_TIMESTAMP.json
```

## Two Evaluation Types

### 1. Stateless Evaluation (`evaluate_router.py`)

Tests single queries without conversation history.

**Categories:**
- **recommendation_queries**: Book recommendations → `recsys`
- **web_search_queries**: Recent books (post-2004) → `web`
- **documentation_queries**: How-to questions → `docs`
- **conversational_queries**: Greetings, thanks → `respond`
- **edge_cases**: Empty, boundary years, gibberish

**30 test queries total**

### 2. History Evaluation (`evaluate_router_with_history.py`)

Tests router with conversation context (tests only final turn).

**Categories:**
- **continuation (easy)**: Same intent maintained - 4 tests
- **easy_switches**: respond/docs → task agents - 6 tests
- **hard_switches**: recsys ↔ web/docs - 6 tests

**16 conversation tests total**

Uses `make_router_input(k_user=2)` to preprocess history exactly as production Conductor does.

## Usage

```bash
# Run from project root

# Stateless evaluation
python evaluation/chatbot/router/evaluate_router.py

# History evaluation
python evaluation/chatbot/router/evaluate_router_with_history.py
```

## Output

Both scripts output:
- Console: Summary with accuracy by category and failed cases
- JSON file: Detailed results in `results/` directory

## Adding Test Cases

**Stateless tests** - edit `router_test_queries.json`:
```json
{
  "recommendation_queries": [
    {"query": "your test query", "expected": "recsys"}
  ]
}
```

**Conversation tests** - edit `router_conversation_tests.json`:
```json
{
  "continuation": [
    {
      "name": "test_name",
      "history": [{"u": "previous query", "a": "previous response"}],
      "final_query": "follow up question",
      "expected": "recsys"
    }
  ]
}
```

## Dependencies

Uses real LLM from `app.agents.settings.get_llm()` - no mocking.
