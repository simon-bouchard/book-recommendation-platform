# Recommendation Agent Evaluation

Comprehensive evaluation suite for the 3-stage recommendation agent pipeline.

## Architecture

The recommendation system uses a **3-stage pipeline**:

```
User Query
    ↓
┌─────────────────┐
│ Planner Agent   │  Analyzes query, determines strategy
└────────┬────────┘
         ↓
┌─────────────────┐
│ Retrieval Agent │  Executes strategy, gathers 60-120 candidates
└────────┬────────┘
         ↓
┌─────────────────┐
│ Curation Agent  │  Ranks, filters, generates prose
└────────┬────────┘
         ↓
   Final Response
```

**Orchestrator** manages the flow between all three stages.

## Evaluation Structure

The evaluation suite contains **35 tests** organized into **9 categories** that test different aspects of the pipeline:

### 1. Planner Evaluation (11 tests)
Tests query analysis and strategy selection:
- **Vague queries** (6 tests): "recommend something good"
  - Warm users (≥10 ratings) → `als_recs`
  - Cold users with profile → `subject_hybrid_pool`
  - Cold users without profile → `popular_books`
- **Descriptive queries** (4 tests): "dark atmospheric mystery with gothic vibes"
  - Should recommend `book_semantic_search`
- **Genre queries** (1 test): "historical fiction"
  - Should recommend `subject_id_search` + `subject_hybrid_pool`

**What's checked:**
- ✅ Correct tool selection for query type
- ✅ Appropriate fallback tools
- ✅ Coherent reasoning
- ✅ Profile data fetched when needed

**Categories:**
- `tool_selection_warm_user`: 6 tests
- `tool_selection_cold_user`: 5 tests

### 2. Retrieval Evaluation (8 tests)
Tests tool execution and strategy adherence:
- **Strategy following**: Does retrieval follow planner's recommendations?
- **Tool arguments**: Correct parameters passed to tools
- **Bug detection**: Critical subject_hybrid_pool argument validation
- **Candidate gathering**: Collects sufficient books (20-30+ minimum)

**What's checked:**
- ✅ Executed recommended tools from strategy
- ✅ Tool arguments are correct (CRITICAL for subject tools)
- ✅ Sufficient candidates gathered
- ✅ Genre queries use explicit subject IDs (not auto-fetch)

**Bug Tests:**
- Genre queries must call `subject_id_search` before `subject_hybrid_pool`
- Genre queries must pass resolved subject IDs, not use auto-fetch
- Vague queries without user subjects should use `popular_books`, not `subject_hybrid_pool`

**Category:**
- `retrieval_strategy_adherence`: 8 tests

### 3. Curation Evaluation (5 tests)
Tests output quality and structure:
- **JSON structure** (3 tests): Valid format, required fields, data types
- **Critical validation** (2 tests): JSON parsing success, inline reference correctness

**What's checked:**
- ✅ Valid response text (>50 chars)
- ✅ Valid book recommendations (≥3 books)
- ✅ Book IDs are integers
- ✅ Inline references match book_ids list
- ✅ No unclosed tags or invalid IDs

**Categories:**
- `curation_quality`: 3 tests
- `curation_critical`: 2 tests

### 4. Negative Constraint Evaluation (2 tests)
Tests handling of negative constraints across pipeline stages:
- **Retrieval exclusion**: "mystery novels but NOT cozy mysteries"
  - Validates retrieval doesn't include "cozy" in search queries
- **Curation filtering**: "thrillers but nothing about serial killers"
  - Uses LLM-as-judge to validate curation filtered constrained books

**What's checked:**
- ✅ Retrieval omits negative terms from queries
- ✅ Curation filters out books matching constraints (LLM-as-judge)
- ✅ Final recommendations don't contain excluded content

**LLM-as-Judge:**
- Uses project's LLM abstraction (reuses agent's LLM instance)
- Cost: ~$0.02 per test (model-dependent)
- Fetches book details from database
- Returns verdict, reasoning, and violating books list

**Category:**
- `negative_constraints`: 2 tests

### 5. Integration Evaluation (9 tests)
Tests full 3-stage pipeline end-to-end:
- **Basic integration** (4 tests): Various query types with different user states
- **High-impact validation** (2 tests): Genre query behavior, personalization override
- **Edge cases** (3 tests): Empty query, very long query, malformed input

**What's checked:**
- ✅ All stages complete successfully
- ✅ Final output has books + prose
- ✅ Tool usage matches expectations
- ✅ Genre queries override personalization (use subjects, not ALS)
- ✅ Graceful error handling

**Categories:**
- `two_stage_integration`: 4 tests
- `integration_high_impact`: 2 tests
- `edge_cases`: 3 tests

## Test Summary

| Category | Tests | Focus |
|----------|-------|-------|
| tool_selection_warm_user | 6 | Warm user tool selection |
| tool_selection_cold_user | 5 | Cold user tool selection |
| retrieval_strategy_adherence | 8 | Tool args + subject bugs |
| curation_quality | 3 | JSON structure basics |
| curation_critical | 2 | JSON parsing + inline refs |
| negative_constraints | 2 | Constraint handling (LLM judge) |
| two_stage_integration | 4 | Basic pipeline flow |
| integration_high_impact | 2 | Genre vs personalization |
| edge_cases | 3 | Error handling |
| **TOTAL** | **35** | |

## Test Users

The evaluation uses specific test users with known characteristics:

- **User 278859**: Warm user (12 ratings) with profile
  - Subjects: Crime (978), Detective (1066), Mystery (2317), Thriller (3248)
- **User 278857**: Cold user (2 ratings) with profile
  - Subjects: Adventure (115), Fantasy (1378)
- **User 278867**: New user (0 ratings), no profile
  - Subjects: None

## Running Tests

### Prerequisites
```bash
# Required environment variable
export DATABASE_URL="postgresql://user:pass@host/db"
```

### Run Evaluation
```bash
cd evaluation/chatbot/recommendation_agent

# Run all tests (default)
python evaluate_recommendation.py

# List available test categories
python evaluate_recommendation.py --list-categories

# Run specific categories
python evaluate_recommendation.py --categories planner
python evaluate_recommendation.py --categories planner retrieval
python evaluate_recommendation.py --categories negative_constraints integration_high_impact

# Output:
# - Console: Real-time test results with pass/fail
# - results/recommendation_eval_YYYYMMDD_HHMMSS.json: Detailed results
```

**Available categories:**
- `tool_selection_warm_user` (6 tests)
- `tool_selection_cold_user` (5 tests)
- `retrieval_strategy_adherence` (8 tests)
- `curation_quality` (3 tests)
- `curation_critical` (2 tests)
- `negative_constraints` (2 tests)
- `two_stage_integration` (4 tests)
- `integration_high_impact` (2 tests)
- `edge_cases` (3 tests)

### Run All Agent Evaluations
```bash
cd evaluation/chatbot
python run_all_evals.py

# Or run specific agents
python run_all_evals.py --agents recommendation router
```

### View Results Dashboard
```bash
cd evaluation/chatbot
python eval_dashboard.py -v
```

## Test Case Format

Test cases are defined in `test_cases.json`:

```json
{
  "tool_selection_warm_user": [
    {
      "name": "vague_query_warm_with_profile",
      "query": "recommend something good for me",
      "user_id": 278859,
      "user_state": {
        "num_ratings": 12,
        "allow_profile": true,
        "is_warm": true
      },
      "expected_tools": {
        "should_use_user_profile": true,
        "profile_before_retrieval": true
      },
      "expected_output": {
        "min_candidates": 20
      }
    }
  ],
  "negative_constraints": [
    {
      "name": "negative_constraint_genre_excluded",
      "query": "recommend mystery novels but NOT cozy mysteries",
      "user_id": 278859,
      "user_state": {
        "num_ratings": 12,
        "allow_profile": false,
        "is_warm": true
      },
      "negative_constraints": ["cozy mystery", "cozy mysteries", "cozy"],
      "expected_retrieval": {
        "should_use_subject_search": true,
        "should_NOT_include_constraints_in_query": true
      },
      "expected_curation": {
        "should_filter_out_cozy_books": true,
        "llm_judge_needed": true,
        "min_final_books": 3
      }
    }
  ]
}
```

## Expected Pass Rates

Target pass rates by evaluation type:

- **Planner**: 85-90% (LLM-based strategy, some variance)
- **Retrieval**: 85-90% (critical bug detection)
- **Curation**: 90-95% (JSON structure validation)
- **Negative Constraints**: 80-85% (LLM-as-judge, complex coordination)
- **Integration**: 85-90% (full pipeline validation)
- **Overall**: **85-90% target**

## Cost Per Run

Approximate costs per full evaluation run:

- **Planner tests** (11): ~$0.11 (uses large model)
- **Retrieval tests** (8): ~$0.12 (uses large model)
- **Curation tests** (5): ~$0.10 (uses large model)
- **Negative constraint tests** (2): ~$0.04 (uses Haiku for judging)
- **Integration tests** (9): ~$0.27 (uses large model)

**Total per run: ~$0.64**

## Results Format

Results are saved to `results/recommendation_eval_TIMESTAMP.json`:

```json
{
  "overall": {
    "passed": 30,
    "total": 35,
    "pass_rate": 0.857
  },
  "eval_type_stats": {
    "planner": {"passed": 10, "total": 11, "pass_rate": 0.909},
    "retrieval": {"passed": 7, "total": 8, "pass_rate": 0.875},
    "curation": {"passed": 5, "total": 5, "pass_rate": 1.000},
    "negative_constraints": {"passed": 2, "total": 2, "pass_rate": 1.000},
    "integration": {"passed": 6, "total": 9, "pass_rate": 0.667}
  },
  "category_stats": {
    "tool_selection_warm_user": {"passed": 5, "total": 6, "pass_rate": 0.833},
    "retrieval_strategy_adherence": {"passed": 7, "total": 8, "pass_rate": 0.875},
    "negative_constraints": {"passed": 2, "total": 2, "pass_rate": 1.000},
    ...
  },
  "results": [...]
}
```

## Database Requirements

Tests require:
- `DATABASE_URL` environment variable set
- Access to users, books, subjects, authors tables
- Semantic search index operational
- Test users (278859, 278857, 278867) exist with known characteristics

## Key Features

✅ **Real agents, real LLM calls**: Tests actual production behavior, not mocks

✅ **Critical bug detection**: Validates subject_hybrid_pool argument handling

✅ **Negative constraint testing**: Uses LLM-as-judge for semantic validation

✅ **Stage separation**: Can test each stage independently or full pipeline

✅ **Genre query validation**: Ensures genre queries override personalization

✅ **Comprehensive coverage**: 35 tests across 4 evaluation types

## Evaluation Philosophy

**Deterministic where possible:**
- Tool selection validation (expected vs actual)
- Argument correctness (type and value checks)
- JSON structure validation
- Inline reference validation

**LLM-as-judge where necessary:**
- Negative constraint filtering (requires semantic understanding)
- Book content matches exclusion criteria

**High-ROI focus:**
- Tests catch frequent, high-impact mistakes
- Critical bugs get multiple test cases
- Cost-effective (cheap Haiku model for judging)

## Troubleshooting

### Common Issues

**"User not found"**
Ensure test users exist in database:
```sql
SELECT user_id, COUNT(*) as ratings
FROM interactions
WHERE user_id IN (278859, 278857, 278867)
GROUP BY user_id;
```

**"No candidates returned"**
Check semantic search index is operational and contains books.

**"Strategy doesn't match expected"**
Planner uses LLM, so some variance is expected (85-90% pass rate is normal).

**"Tool arguments incorrect" (CRITICAL)**
Check retrieval agent passes explicit subject IDs for genre queries, not relying on auto-fetch.

**"Negative constraint test failed"**
Review judge reasoning in results JSON. May indicate curation agent not filtering correctly, or LLM judge may have failed (check error details).

## Adding New Tests

1. Add test case to `test_cases.json` in appropriate category
2. Specify user_id (must exist in database)
3. Define user_state (num_ratings, allow_profile, is_warm)
4. Set expected_tools, expected_output, or expected_behavior
5. For negative constraints, set negative_constraints list and llm_judge_needed
6. Run evaluation to validate

Example:
```json
{
  "name": "new_test_case",
  "query": "recommend science fiction",
  "user_id": 278859,
  "user_state": {
    "num_ratings": 12,
    "allow_profile": false,
    "is_warm": true
  },
  "expected_tools": {
    "should_use_subject_search": true
  },
  "expected_output": {
    "min_candidates": 20
  }
}
```

## Integration with Dashboard

Results are compatible with `eval_dashboard.py`:

```bash
# View all agent results
python eval_dashboard.py -v

# Should show:
# recommendation    85.7%    30/35    ✅ Good
```

## Related Testing

- **Pytest Integration Tests**: `tests/integration/chatbot/agents/recommendation/`
  - Tests orchestration infrastructure with mocked agents (fast, cheap)
  - Validates data flow, parameter propagation, error handling
  - 24 tests, ~10 seconds runtime

- **Evaluation Tests** (this suite): `evaluation/chatbot/recommendation_agent/`
  - Tests real agent behavior with actual LLM calls
  - Validates tool selection, argument correctness, output quality
  - 35 tests, ~5-10 minutes runtime, ~$0.64 per run
