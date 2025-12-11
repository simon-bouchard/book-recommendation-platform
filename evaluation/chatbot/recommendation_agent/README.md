# Recommendation Agent Evaluation

Comprehensive evaluation suite for the 3-stage recommendation agent pipeline with isolated stage testing.

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

## Evaluation Approach

The evaluation suite uses **isolated stage testing** where possible:

- **Planner tests**: Query → Planner → Validate strategy (full planner execution)
- **Retrieval tests**: Mock strategy → Retrieval → Validate execution (no planner call)
- **Curation tests**: Mock candidates + context → Curation → Validate output (no retrieval call)
- **Integration tests**: Full pipeline → Validate coordination (all 3 stages)

### Benefits of Isolated Testing

- **Faster**: Fewer LLM calls per test (saves planner/retrieval costs for curation tests)
- **Cheaper**: ~30% cost reduction per run
- **Better debugging**: Clear failure attribution (know exactly which stage broke)
- **Preserved coordination**: Integration tests still validate full pipeline behavior

### Test Data Factory

Tests use a **test data factory** that provides:
- Mock `PlannerStrategy` objects for retrieval tests
- Real tool queries for candidate books (semantic_search, subject_hybrid_pool, als_recs)
- Mock `ExecutionContext` objects for curation tests
- Automatic shuffling for mixed candidate tests

### LLM-as-Judge

Complex semantic validation uses **LLM-as-judge** instead of brittle keyword matching:
- Genre matching validation
- Personalization prose quality validation
- Negative constraint filtering validation
- False personalization detection

## Evaluation Structure

The evaluation suite contains **41 tests** organized into **12 categories** that test different aspects of the pipeline:

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
- Correct tool selection for query type
- Appropriate fallback tools
- Coherent reasoning
- Profile data fetched when needed

**Testing approach:** Full planner execution

**Categories:**
- `tool_selection_warm_user`: 6 tests
- `tool_selection_cold_user`: 5 tests

### 2. Retrieval Evaluation (8 tests)
Tests tool execution and strategy adherence using **isolated testing**:
- **Strategy following**: Does retrieval follow mock strategy recommendations?
- **Tool arguments**: Correct parameters passed to tools
- **Bug detection**: Critical subject_hybrid_pool argument validation
- **Candidate gathering**: Collects sufficient books (20-30+ minimum)

**What's checked:**
- Executed recommended tools from strategy
- Tool arguments are correct (CRITICAL for subject tools)
- Sufficient candidates gathered
- Genre queries use explicit subject IDs (not auto-fetch)

**Testing approach:** Mock strategy → Retrieval (no planner call)

**Bug Tests:**
- Genre queries must call `subject_id_search` before `subject_hybrid_pool`
- Genre queries must pass resolved subject IDs, not use auto-fetch
- Vague queries without user subjects should use `popular_books`, not `subject_hybrid_pool`

**Category:**
- `retrieval_strategy_adherence`: 8 tests

### 3. Curation Evaluation (11 tests)
Tests output quality and structure using **isolated testing**:

**Basic Structure (5 tests):**
- **JSON structure** (3 tests): Valid format, required fields, data types
- **Critical validation** (2 tests): JSON parsing success, inline reference correctness

**What's checked:**
- Valid response text (>50 chars)
- Valid book recommendations (≥3 books)
- Book IDs are integers
- Inline references match book_ids list
- No unclosed tags or invalid IDs

**Testing approach:** Mock candidates + context → Curation (no full pipeline)

**Genre Matching (2 tests):**
- **Fantasy filtering**: 40 fantasy + 20 mystery/thriller → filter wrong genre
- **Historical fiction filtering**: 40 historical + 20 sci-fi/fantasy → filter wrong genre

**What's checked:**
- Curation filters out wrong-genre books from mixed pool (LLM-as-judge)
- At least 80% of final books match expected genre
- Shuffled candidates ensure real filtering (not just taking first N)

**Testing approach:** Mixed candidates (shuffled) + context → Curation → LLM judge

**Personalization Prose (3 tests):**
- **ALS prose**: Should mention reading history/personalization
- **Favorite subjects prose**: Should reference user's interests/genres
- **Recent interactions prose**: Should acknowledge recent reads

**What's checked:**
- Prose correctly reflects personalization method (LLM-as-judge)
- ALS context → mentions reading history
- Profile context → mentions favorite genres/interests
- No false claims when no personalization used

**Testing approach:** Candidates + personalization context → Curation → LLM judge

**False Personalization (1 test):**
- **No false claims**: Prose should NOT claim personalization when none used

**What's checked:**
- Prose doesn't falsely mention reading history/preferences (LLM-as-judge)
- Focus on query matching, not user history

**Testing approach:** Non-personalized context → Curation → LLM judge validates absence of claims

**Categories:**
- `curation_quality`: 3 tests (basic structure)
- `curation_critical`: 2 tests (parsing and references)
- `curation_genre_matching`: 2 tests (genre filtering)
- `curation_personalization_prose`: 3 tests (prose quality)
- `curation_false_personalization`: 1 test (false claim detection)

### 4. Negative Constraint Evaluation (2 tests)
Tests handling of negative constraints across pipeline stages:
- **Retrieval exclusion**: "mystery novels but NOT cozy mysteries"
  - Validates retrieval doesn't include "cozy" in search queries
- **Curation filtering**: "thrillers but nothing about serial killers"
  - Uses LLM-as-judge to validate curation filtered constrained books

**What's checked:**
- Retrieval omits negative terms from queries
- Curation filters out books matching constraints (LLM-as-judge)
- Final recommendations don't contain excluded content

**LLM-as-Judge:**
- Uses project's LLM abstraction (reuses agent's LLM instance)
- Cost: ~$0.005 per test (uses efficient model)
- Fetches book details from database
- Returns verdict, reasoning, and violating books list

**Testing approach:** Full pipeline (to test both retrieval and curation coordination)

**Category:**
- `negative_constraints`: 2 tests

### 5. Integration Evaluation (9 tests)
Tests full 3-stage pipeline end-to-end:
- **Basic integration** (4 tests): Various query types with different user states
- **High-impact validation** (2 tests): Genre query behavior, personalization override
- **Edge cases** (3 tests): Empty query, very long query, malformed input

**What's checked:**
- All stages complete successfully
- Final output has books + prose
- Tool usage matches expectations
- Genre queries override personalization (use subjects, not ALS)
- Graceful error handling

**Testing approach:** Full pipeline (Query → Planner → Retrieval → Curation)

**Categories:**
- `two_stage_integration`: 4 tests
- `integration_high_impact`: 2 tests
- `edge_cases`: 3 tests

## Test Summary

| Category | Tests | Approach | Focus |
|----------|-------|----------|-------|
| tool_selection_warm_user | 6 | Planner only | Warm user tool selection |
| tool_selection_cold_user | 5 | Planner only | Cold user tool selection |
| retrieval_strategy_adherence | 8 | Mock strategy → Retrieval | Tool args + subject bugs |
| curation_quality | 3 | Mock candidates → Curation | JSON structure basics |
| curation_critical | 2 | Mock candidates → Curation | JSON parsing + inline refs |
| curation_genre_matching | 2 | Mixed candidates → Curation | Genre filtering (LLM judge) |
| curation_personalization_prose | 3 | Candidates + context → Curation | Prose quality (LLM judge) |
| curation_false_personalization | 1 | Non-personalized → Curation | False claim detection (LLM judge) |
| negative_constraints | 2 | Full pipeline | Constraint handling (LLM judge) |
| two_stage_integration | 4 | Full pipeline | Basic pipeline flow |
| integration_high_impact | 2 | Full pipeline | Genre vs personalization |
| edge_cases | 3 | Full pipeline | Error handling |
| **TOTAL** | **41** | | |

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

# Run new tests
python evaluate_recommendation.py --categories curation_genre_matching
python evaluate_recommendation.py --categories curation_personalization_prose curation_false_personalization

# Output:
# - Console: Real-time test results with pass/fail
# - results/recommendation_eval_YYYYMMDD_HHMMSS.json: Detailed results
```

**Available categories:**
- `tool_selection_warm_user` (6 tests) - Planner for warm users
- `tool_selection_cold_user` (5 tests) - Planner for cold users
- `retrieval_strategy_adherence` (8 tests) - Retrieval execution
- `curation_quality` (3 tests) - Basic curation structure
- `curation_critical` (2 tests) - Critical curation validation
- `curation_genre_matching` (2 tests) - Genre filtering
- `curation_personalization_prose` (3 tests) - Personalization prose quality
- `curation_false_personalization` (1 test) - False claim detection
- `negative_constraints` (2 tests) - Constraint handling
- `two_stage_integration` (4 tests) - Basic integration
- `integration_high_impact` (2 tests) - Critical integration scenarios
- `edge_cases` (3 tests) - Error handling

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

## Expected Pass Rates

Target pass rates by evaluation type:

- **Planner**: 85-90% (LLM-based strategy, some variance)
- **Retrieval**: 85-90% (critical bug detection)
- **Curation (structure)**: 90-95% (JSON structure validation)
- **Curation (genre)**: 80-85% (LLM-as-judge, complex filtering)
- **Curation (prose)**: 85-90% (LLM-as-judge, semantic validation)
- **Negative Constraints**: 80-85% (LLM-as-judge, complex coordination)
- **Integration**: 85-90% (full pipeline validation)
- **Overall**: **85-90% target**

## Cost Per Run

Approximate costs per full evaluation run (41 tests):

- **Planner tests** (11): ~$0.11 (uses large model, full execution)
- **Retrieval tests** (8): ~$0.08 (uses large model, NO planner call - saves ~$0.04)
- **Curation tests** (11): ~$0.12 (uses large model, NO retrieval - saves ~$0.08)
  - Basic structure (5): ~$0.05
  - Genre matching (2): ~$0.03 (includes LLM judge)
  - Personalization prose (4): ~$0.04 (includes LLM judge)
- **Negative constraint tests** (2): ~$0.04 (uses efficient model for judging)
- **Integration tests** (9): ~$0.27 (uses large model, full pipeline)

**Total per run: ~$0.45-0.50** (vs. ~$0.64 before refactoring)

**Savings: 22-30% per run**

## Results Format

Results are saved to `results/recommendation_eval_TIMESTAMP.json`:

```json
{
  "overall": {
    "passed": 36,
    "total": 41,
    "pass_rate": 0.878
  },
  "eval_type_stats": {
    "planner": {"passed": 10, "total": 11, "pass_rate": 0.909},
    "retrieval": {"passed": 7, "total": 8, "pass_rate": 0.875},
    "curation": {"passed": 5, "total": 5, "pass_rate": 1.000},
    "curation_genre": {"passed": 2, "total": 2, "pass_rate": 1.000},
    "curation_prose": {"passed": 3, "total": 4, "pass_rate": 0.750},
    "negative_constraints": {"passed": 2, "total": 2, "pass_rate": 1.000},
    "integration": {"passed": 7, "total": 9, "pass_rate": 0.778}
  },
  "category_stats": {
    "tool_selection_warm_user": {"passed": 5, "total": 6, "pass_rate": 0.833},
    "retrieval_strategy_adherence": {"passed": 7, "total": 8, "pass_rate": 0.875},
    "curation_genre_matching": {"passed": 2, "total": 2, "pass_rate": 1.000},
    "curation_personalization_prose": {"passed": 2, "total": 3, "pass_rate": 0.667},
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

**Real agents, real LLM calls**: Tests actual production behavior, not mocks

**Isolated stage testing**: Each stage can be tested independently with controlled inputs

**Critical bug detection**: Validates subject_hybrid_pool argument handling

**LLM-as-judge validation**: Semantic validation for genre matching, prose quality, constraint handling

**Stage separation**: Can test each stage independently or full pipeline

**Genre query validation**: Ensures genre queries override personalization

**Comprehensive coverage**: 41 tests across 5 evaluation types

**Test data factory**: Centralized mock data generation with realistic tool queries

**Shuffled candidates**: Forces actual filtering logic (prevents accidental passes)

## Evaluation Philosophy

**Deterministic where possible:**
- Tool selection validation (expected vs actual)
- Argument correctness (type and value checks)
- JSON structure validation
- Inline reference validation

**LLM-as-judge where necessary:**
- Genre matching (requires semantic understanding)
- Personalization prose quality (requires natural language understanding)
- Negative constraint filtering (requires content understanding)
- False claim detection (requires semantic analysis)

**Isolated testing benefits:**
- Faster execution (fewer LLM calls)
- Cheaper costs (skip unnecessary stages)
- Better debugging (clear failure attribution)
- Preserved integration validation (separate integration tests)

**High-ROI focus:**
- Tests catch frequent, high-impact mistakes
- Critical bugs get multiple test cases
- Cost-effective (efficient models for judging)

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

**"Strategy doesn't match expected"** (Planner tests)
Planner uses LLM, so some variance is expected (85-90% pass rate is normal).

**"Tool arguments incorrect" (CRITICAL)** (Retrieval tests)
Check retrieval agent passes explicit subject IDs for genre queries, not relying on auto-fetch.

**"Genre matching failed"** (Curation genre tests)
Review LLM judge reasoning in results JSON. May indicate curation not filtering correctly, or edge case genre overlap.

**"Prose validation failed"** (Curation prose tests)
Review LLM judge reasoning. Check execution context matches expected personalization method.

**"Negative constraint test failed"**
Review judge reasoning in results JSON. May indicate curation agent not filtering correctly, or LLM judge may have failed (check error details).

## Adding New Tests

1. Add test case to `test_cases.json` in appropriate category
2. Specify user_id (must exist in database)
3. Define user_state (num_ratings, allow_profile, is_warm)
4. Set expected_tools, expected_output, or expected_behavior
5. For negative constraints or genre matching, set llm_judge_needed
6. For personalization prose tests, specify test_scenario and context_scenario
7. Run evaluation to validate

Example (curation genre test):
```json
{
  "name": "genre_match_science_fiction",
  "query": "recommend science fiction",
  "user_id": 278859,
  "user_state": {
    "num_ratings": 12,
    "allow_profile": false,
    "is_warm": true
  },
  "test_scenario": "genre_scifi",
  "expected_genre": "science fiction",
  "expected_curation": {
    "should_filter_wrong_genre": true,
    "llm_judge_needed": true,
    "min_final_books": 3,
    "min_genre_match_rate": 0.8
  }
}
```

Then add corresponding scenario to `test_data_factory.py` if needed.

## Integration with Dashboard

Results are compatible with `eval_dashboard.py`:

```bash
# View all agent results
python eval_dashboard.py -v

# Should show:
# recommendation    87.8%    36/41    Good
```

## Related Testing

- **Pytest Integration Tests**: `tests/integration/chatbot/agents/recommendation/`
  - Tests orchestration infrastructure with mocked agents (fast, cheap)
  - Validates data flow, parameter propagation, error handling
  - 24 tests, ~10 seconds runtime

- **Evaluation Tests** (this suite): `evaluation/chatbot/recommendation_agent/`
  - Tests real agent behavior with actual LLM calls
  - Validates tool selection, argument correctness, output quality
  - 41 tests, ~5-10 minutes runtime, ~$0.45-0.50 per run

## Test Data Factory

Tests use a centralized factory for mock data:

**Location**: `test_data_factory.py`

**Functions:**
- `get_mock_strategy(scenario)` - Returns PlannerStrategy for retrieval tests
- `get_candidates(scenario, db)` - Returns candidate books for curation tests
- `get_execution_context(scenario)` - Returns ExecutionContext for curation tests

**Scenarios:**
- Strategies: semantic, als, subject, profile, negative, fallback
- Candidates: basic, negative_cozy, negative_serial_killer, genre_fantasy, genre_historical, als, subject
- Contexts: semantic, als, subject, profile, profile_recent, negative, fallback, no_personalization

## LLM Judges

**Location**: `llm_judges.py`

**Functions:**
- `llm_judge_genre_match(books, expected_genre, db, judge_llm)` - Validates genre accuracy
- `llm_judge_personalization_prose(response_text, execution_context, judge_llm)` - Validates prose quality

**Features:**
- Uses project's LLM abstraction (reuses agent's LLM instance)
- Returns detailed verdict with reasoning
- Cost-effective (efficient models)
- Flexible validation criteria

## Query Validation

Before running evals, validate queries return expected book types:

```bash
python validate_queries.py
```

This verifies:
- Negative constraint queries return constrained books
- Genre queries return correct genre books
- Minimal overlap between base and constraint queries
