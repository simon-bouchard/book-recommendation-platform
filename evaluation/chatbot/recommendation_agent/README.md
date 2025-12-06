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

The evaluation is organized into **4 test types** that correspond to the architecture:

### 1. Planner Evaluation (18 tests)
Tests query analysis and strategy selection:
- **Vague queries**: "recommend something good"
  - Warm users (≥10 ratings) → `als_recs`
  - Cold users with profile → `subject_hybrid_pool`
  - Cold users without profile → `popular_books`
- **Descriptive queries**: "dark atmospheric mystery with gothic vibes"
  - Should recommend `book_semantic_search`
- **Genre queries**: "historical fiction"
  - Should recommend `subject_id_search` + `subject_hybrid_pool`

**What's checked:**
- ✅ Correct tool selection for query type
- ✅ Appropriate fallback tools
- ✅ Coherent reasoning
- ✅ Profile data fetched when needed

### 2. Retrieval Evaluation (15 tests)
Tests tool execution and strategy following:
- **Strategy adherence**: Does retrieval follow planner's recommendations?
- **Tool arguments**: Correct parameters (top_k, query strings, subject IDs)
- **Candidate gathering**: Collects 60-120 books minimum
- **Fallback logic**: Uses backup tools when primary underperforms

**What's checked:**
- ✅ Executed recommended tools from strategy
- ✅ Tool arguments are correct
- ✅ Sufficient candidates gathered (≥20 minimum)
- ✅ Appropriate fallback behavior

### 3. Curation Evaluation (10 tests)
Tests output quality and structure:
- **JSON validity**: Proper structure returned
- **Required fields**: `book_ids`, `response_text`, `reasoning`
- **Inline references**: Valid `<book id="123">` tags
- **Book ordering**: Preserves LLM's ranking

**What's checked:**
- ✅ Valid response text (>50 chars)
- ✅ Valid book recommendations (≥3 books)
- ✅ Book IDs are integers
- ✅ Inline references match book_ids list

### 4. Integration Evaluation (8 tests)
Tests full 3-stage pipeline end-to-end:
- **Happy paths**: Various query types with different user states
- **Edge cases**: Empty query, very long query, no candidates
- **Data flow**: Context passes correctly between stages

**What's checked:**
- ✅ All stages complete successfully
- ✅ Final output has books + prose
- ✅ Graceful error handling
- ✅ Appropriate fallback behavior

## Test Users

The evaluation uses specific test users with known characteristics:

- **User 278859**: Warm user (12 ratings) with profile
  - Subjects: Detective/Mystery, Thriller, Crime
- **User 278857**: Cold user (1 rating) with profile
  - Subjects: Fantasy, Adventure
- **User 278867**: New user (0 ratings), no profile

## Running Tests

### Run Recommendation Agent Only
```bash
cd evaluation/chatbot/recommendation_agent
python evaluate_recommendation.py
```

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
  ]
}
```

## Expected Pass Rates

Based on the evaluation plan:

- **Planner**: 85-95% (deterministic strategy checks)
- **Retrieval**: 80-90% (tool execution validation)
- **Curation**: 85-95% (JSON structure checks)
- **Integration**: 80-90% (depends on data quality)
- **Overall**: 85-90% target

## Results

Results are saved to `results/recommendation_eval_TIMESTAMP.json` with structure:

```json
{
  "overall": {
    "passed": 45,
    "total": 51,
    "pass_rate": 0.882
  },
  "eval_type_stats": {
    "planner": {"passed": 16, "total": 18, "pass_rate": 0.889},
    "retrieval": {"passed": 13, "total": 15, "pass_rate": 0.867},
    "curation": {"passed": 9, "total": 10, "pass_rate": 0.900},
    "integration": {"passed": 7, "total": 8, "pass_rate": 0.875}
  },
  "results": [...]
}
```

## Database Requirements

Tests require:
- `DATABASE_URL` environment variable set
- Access to users, books, subjects tables
- Semantic search index operational
- Test users (278859, 278857, 278867) exist

## Key Design Decisions

✅ **Real tools, no mocking**: Tools are fast and give realistic validation

✅ **Real database**: Tests execute against actual data

✅ **Deterministic checks**: No LLM-as-judge for tool selection (faster, cheaper)

✅ **Stage separation**: Can test each stage independently or full pipeline

✅ **Reusable test users**: Known user states enable consistent testing

## Troubleshooting

### Common Issues

**"User not found"**: Ensure test users exist in database
```sql
SELECT user_id, COUNT(*) as ratings
FROM interactions
WHERE user_id IN (278859, 278857, 278867)
GROUP BY user_id;
```

**"No candidates returned"**: Check semantic search index is operational

**"Strategy doesn't match expected"**: Planner uses LLM, so some variance is normal (85-95% pass rate expected)

**"Tool arguments incorrect"**: Check retrieval agent is reading strategy correctly

## Adding New Tests

1. Add test case to `test_cases.json` in appropriate category
2. Specify user_id (must exist in database)
3. Define user_state (num_ratings, allow_profile, is_warm)
4. Set expected_tools or expected_behavior
5. Run evaluation to validate

## Integration with Dashboard

Results are compatible with `eval_dashboard.py`:

```bash
# View all agent results
python eval_dashboard.py -v

# Should show:
# recommendation    85.0%    43/51    ✅ Good
```
