# Web Agent Evaluation

Simplified two-layer evaluation to verify Web Agent correctly uses web search and provides relevant, current information.

## Structure

```
evaluation/chatbot/web_agent/
├── evaluate_web.py         # Evaluation script
├── web_test_cases.json     # 8 test cases
└── results/                # Output (auto-created)
```

## Usage

```bash
python evaluation/chatbot/web_agent/evaluate_web.py
```

## Two-Layer Testing

### Layer 1: Tool Usage (Deterministic)
Validates agent uses web_search tool.

**Checks:**
- Did agent call `web_search` tool?

**Pass criteria:** web_search was called

### Layer 2: Answer Quality (LLM-as-judge)
Validates answer relevance and recency.

**Scoring (binary 0/1):**
- **Relevance**: Does response address the query?
- **Recency**: Does response contain current/recent information? (only checked when `recency_expected: true`)

**Pass criteria:** Both scores = 1

## Overall Pass

All three must be true:
1. Agent executed successfully (no errors)
2. Layer 1 passed (web_search used)
3. Layer 2 passed (quality scores = 1)

## Test Cases

8 queries covering:
- Recent book releases (2024-2025) - 6 tests with recency checking
- General book information - 2 tests without recency checking

Each test case includes:
- `query`: User question
- `recency_expected`: Whether to check for current information

## Why Simplified?

**Web results are non-deterministic:**
- Can't validate specific sources (change over time)
- Can't check exact keywords (too brittle)
- Can't provide ground truth (web content varies)

**Focus on behavior:**
- Did agent use web tools? (Layer 1)
- Is answer relevant and current? (Layer 2)

## Output

Console shows:
- Overall pass rate
- Layer 1 (tool usage) pass rate
- Layer 2 (quality) pass rate
- Per-test results with both layers

JSON saved to `results/` with full details.

## Success Criteria

- Overall pass rate: ≥75% (6/8 tests)
- Layer 1 pass rate: ≥85%
- Layer 2 pass rate: ≥75%

## Notes

**Lower thresholds than other agents** because:
- Web results are non-deterministic
- Recency checking depends on current web content
- External API availability may vary

**Recency checking** only applies to queries explicitly asking about recent/new books. General queries about authors or book series don't require recency.
