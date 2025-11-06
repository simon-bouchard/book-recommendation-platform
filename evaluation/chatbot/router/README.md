# Router Evaluation

Tests RouterLLM classification accuracy with real LLM queries.

## Structure

```
evaluation/chatbot/router/
├── README.md                      # This file
├── router_test_queries.json       # Test cases
├── evaluate_router.py             # Main evaluation script
└── results/                       # Output directory (auto-created)
    └── router_eval_TIMESTAMP.json
```

## Test Categories

- **recommendation_queries**: Book recommendation requests → `recsys`
- **web_search_queries**: Recent books (post-2004) → `web`
- **documentation_queries**: How-to questions → `docs`
- **conversational_queries**: Greetings, thanks → `respond`
- **edge_cases**: Empty, boundary years, gibberish

## Usage

```bash
# Run from project root
python evaluation/router/evaluate_router.py
```

## Output

- Console: Summary with accuracy by category and failed cases
- JSON file: Detailed results in `results/` directory

## Adding Test Cases

Edit `router_test_queries.json`:

```json
{
  "recommendation_queries": [
    {"query": "your test query", "expected": "recsys"}
  ]
}
```

## Dependencies

Uses real LLM from `app.agents.settings.get_llm()` - no mocking.
