# RecommendationAgent Pipeline Integration Tests

Integration tests for the RecommendationAgent's three-stage pipeline using mocked sub-agents.

## Purpose

These tests validate the RecommendationAgent orchestration logic **without making LLM calls**:

- ✅ Data flows correctly through Planner → Retrieval → Curation stages
- ✅ Parameters propagate to all sub-agents
- ✅ ExecutionContext is assembled properly
- ✅ Error handling works at each stage
- ✅ Complete pipeline works end-to-end

**What these tests DON'T cover** (tested elsewhere):
- ❌ Planner tool selection accuracy → Component tests
- ❌ Retrieval tool execution quality → Component tests
- ❌ Curation ranking quality → Evaluation tests
- ❌ LLM response quality → Evaluation tests

## Testing Strategy

### Mock-Based Pipeline Testing

All tests use **mocked sub-agents** instead of real LLM agents:

```python
def test_something(mock_planner_builder, mock_retrieval_builder, mock_curation_builder):
    # Configure mocks with builders
    mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
    mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
    mock_curation = mock_curation_builder.returns_success_with_books(5).build()

    # Inject mocks into orchestrator
    agent = RecommendationAgent(
        planner_agent=mock_planner,
        retrieval_agent=mock_retrieval,
        curation_agent=mock_curation,
        # ... other params
    )

    result = agent.execute(request)

    # Verify data flow (not just calls)
    retrieval_input = mock_retrieval.execute.call_args[0][0]
    assert retrieval_input.strategy == expected_strategy
```

**Why mocked sub-agents?**
- ⚡ **Fast**: <10 seconds for all 24 tests (vs 5-10 minutes with real LLMs)
- 💰 **Free**: No API costs
- ✅ **Deterministic**: Same results every time (no LLM variance)
- 🎯 **Focused**: Tests orchestration, not LLM behavior
- 🔄 **CI-friendly**: Can run on every commit

### Test Organization

```
tests/integration/chatbot/agents/recommendation/
├── conftest.py                      # Mock fixtures and builders
└── test_pipeline_integration.py    # 24 tests organized by category
    ├── TestStageTransitions         # 6 tests - Data flow between stages
    ├── TestParameterPropagation     # 4 tests - Parameter passing
    ├── TestErrorHandling            # 8 tests - Failure scenarios
    └── TestFullPipelineFlow         # 6 tests - End-to-end scenarios
```

**Total: 24 tests**

## Test Categories

### 1. Stage Transitions (6 tests)
Tests that data flows correctly between pipeline stages:

- **test_planner_strategy_reaches_retrieval**: PlannerStrategy → RetrievalInput
- **test_retrieval_candidates_reach_curation**: Candidates → CurationAgent
- **test_execution_context_assembled_correctly**: ExecutionContext has all fields
- **test_profile_data_flows_through_all_stages**: Profile data preserved
- **test_candidate_metadata_preserved**: Book metadata survives pipeline
- **test_book_ids_survive_full_pipeline**: Final output has correct book IDs

**Why critical:** Silent data loss between stages causes incorrect recommendations.

### 2. Parameter Propagation (4 tests)
Tests that user context reaches all sub-agents:

- **test_cold_user_parameters_reach_all_stages**: num_ratings < 10 propagates
- **test_warm_user_parameters_reach_all_stages**: num_ratings >= 10 propagates
- **test_profile_access_flag_propagates**: allow_profile flag respected (privacy)
- **test_optional_parameters_handled**: None values handled gracefully

**Why critical:** Parameters control agent behavior. Incorrect propagation breaks personalization.

### 3. Error Handling (8 tests)
Tests failure scenarios at each stage:

- **test_planner_failure_uses_fallback_strategy**: Planner error → hardcoded fallback
- **test_retrieval_failure_returns_error_response**: No candidates → helpful error
- **test_curation_failure_returns_fallback_response**: Curation error → simple list
- **test_zero_candidates_from_retrieval_handled**: Retrieval returns empty list gracefully
- **test_curation_returns_empty_books_handled**: Curation filters out all books gracefully
- **test_orchestrator_timeout_boundary**: Pipeline respects timeout constraints
- **test_database_none_handled_gracefully**: Missing db doesn't crash
- **test_invalid_input_doesnt_crash**: Empty query, malformed input handled

**Why critical:** Production agents fail. System must degrade gracefully.

### 4. Full Pipeline Flow (6 tests)
Tests complete end-to-end execution:

- **test_warm_user_vague_query_complete_flow**: Warm user + ALS recommendations
- **test_cold_user_descriptive_query_complete_flow**: Cold user + semantic search
- **test_cold_user_with_profile_complete_flow**: Cold user + profile + subjects
- **test_empty_query_handled**: Empty query doesn't crash
- **test_very_long_query_handled**: Very long query doesn't crash
- **test_first_turn_no_history_through_pipeline**: First turn with no history handled correctly

**Why critical:** Validates that all stages work together correctly.

## Running Tests

### Prerequisites

1. **Database connection required:**
   ```bash
   export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
   ```

2. **Test users must exist:**
   - User 278859: warm user (≥10 ratings)
   - User 278857: cold user (<10 ratings)
   - User with profile data

   Tests will attempt to use these specific users, or fall back to any users matching the criteria.

### Run All Pipeline Tests

```bash
# From project root
pytest tests/integration/chatbot/agents/recommendation/ -v

# Expected: 24 passed
# Expected time: <10 seconds
```

### Run Specific Category

```bash
# Run stage transition tests only
pytest tests/integration/chatbot/agents/recommendation/test_pipeline_integration.py::TestStageTransitions -v

# Run error handling tests only
pytest tests/integration/chatbot/agents/recommendation/test_pipeline_integration.py::TestErrorHandling -v
```

### Run Single Test

```bash
pytest tests/integration/chatbot/agents/recommendation/test_pipeline_integration.py::TestStageTransitions::test_planner_strategy_reaches_retrieval -v
```

## Fixtures

### Generic Fixtures (from parent conftest.py)
- `db_session`: Database session for all tests

### Recsys Fixtures (from recsys_fixtures.py)
- `test_user_warm`: User with ≥10 ratings
- `test_user_cold`: User with <10 ratings
- `test_user_new`: User with 0 ratings
- `test_user_with_profile`: User with favorite subjects

### Mock Builders (from recommendation/conftest.py)
- `mock_planner_builder`: Configure PlannerAgent behavior
- `mock_retrieval_builder`: Configure RetrievalAgent behavior
- `mock_curation_builder`: Configure CurationAgent behavior

### Test Data Factories (from recommendation/conftest.py)
- `candidate_factory`: Generate test book candidates
- `strategy_factory`: Generate test PlannerStrategy objects

## Expected Behavior

### Success Criteria
- **All 24 tests pass** on fresh database with test users
- **Execution time**: <10 seconds total
- **No flaky tests**: deterministic results (mocked responses)
- **No API calls**: $0.00 cost per run

### Common Failures

**"Mock not called" errors:**
- Mock wasn't injected properly
- Solution: Verify sub-agents are passed to RecommendationAgent constructor

**"Attribute not found on mock" errors:**
- Mock return value doesn't match expected schema
- Solution: Check builder configuration in conftest.py

**Test fails but should pass:**
- Orchestrator logic changed but tests not updated
- Solution: Update test assertions to match new behavior

## Key Design Patterns

### Pattern 1: Use Builders for Configuration
```python
# Good - fluent API
mock_planner = mock_planner_builder.returns_warm_user_strategy().build()

# Bad - manual mock configuration
mock_planner = Mock()
mock_planner.execute.return_value = ...  # Repetitive and error-prone
```

### Pattern 2: Verify Data Flow (Not Just Calls)
```python
# Good - verify what was passed
retrieval_input = mock_retrieval.execute.call_args[0][0]
assert retrieval_input.strategy == expected_strategy

# Bad - only verify it was called
assert mock_retrieval.execute.called
```

### Pattern 3: Inject Mocks via Constructor
```python
# Good - dependency injection
agent = RecommendationAgent(
    planner_agent=mock_planner,
    retrieval_agent=mock_retrieval,
    curation_agent=mock_curation,
)

# Bad - monkey patching
agent = RecommendationAgent()
agent.planner_agent = mock_planner  # May not work correctly
```

## Maintenance

### Adding New Tests
1. Choose appropriate test class based on category
2. Use existing builders (`mock_planner_builder`, etc.)
3. Follow naming convention: `test_<what>_<condition>`
4. Add docstring explaining what's tested
5. Verify data flow, not just mock calls

### Updating for Code Changes
- **Planner changes**: Update mock planner responses in conftest.py
- **Retrieval changes**: Update mock retrieval responses in conftest.py
- **Curation changes**: Update mock curation responses in conftest.py
- **New sub-agents**: Add new builder class to conftest.py
- **Schema changes**: Update factory methods

## Architecture Context

### RecommendationAgent Pipeline

The RecommendationAgent orchestrates a three-stage pipeline:

```
Stage 1: PlannerAgent
├─ Input: PlannerInput (query, has_als_recs, allow_profile, available_tools)
├─ Process: Analyze query, determine strategy
└─ Output: PlannerStrategy (recommended_tools, fallback_tools, reasoning, profile_data)

Stage 2: RetrievalAgent
├─ Input: RetrievalInput (query, strategy, profile_data)
├─ Process: Execute tools, gather 60-120 candidates
└─ Output: (candidates: List[BookRecommendation], tool_executions: List[ToolExecution])

Stage 3: CurationAgent
├─ Input: (request, candidates, execution_context)
├─ Process: Rank, filter, generate prose
└─ Output: AgentResponse (text, book_recommendations, success, policy_version)
```

### What Gets Tested Here

**Infrastructure concerns** (these tests):
- Does strategy reach retrieval correctly?
- Do candidates reach curation correctly?
- Does ExecutionContext have all fields?
- Do parameters propagate through all stages?
- Does error handling work?

**Quality concerns** (evaluation tests, separate):
- Does planner pick the right tools?
- Do retrieved candidates match the query?
- Does curation rank books correctly?
- Is the prose response helpful?

## Related Testing

- **Conductor Tests**: `tests/integration/chatbot/conductor/` - Tests agent routing
- **Component Tests**: (if added) - Test individual sub-agents
- **Evaluation Tests**: `tests/evaluation/chatbot/` - Test LLM quality

---

**For questions about these tests, refer to the implementation plan in Phase 2**
