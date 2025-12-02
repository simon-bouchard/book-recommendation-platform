# Conductor Integration Tests

Integration tests for the Conductor orchestration layer using mocked agents.

## Purpose

These tests validate Conductor's orchestration logic **without making LLM calls**:

- ✅ Routing decisions are executed correctly
- ✅ Data flows through adapter conversions (TurnInput ↔ AgentRequest ↔ AgentResult)
- ✅ Context builders truncate history correctly
- ✅ Parameters propagate through all layers
- ✅ Multi-turn conversation state is managed properly
- ✅ Errors are handled gracefully

**What these tests DON'T cover** (tested elsewhere):
- ❌ Router classification accuracy → `evaluation/chatbot/router/`
- ❌ Agent tool selection logic → Agent-specific tests
- ❌ LLM response quality → Evaluation tests

## Testing Strategy

### Mock-Based Infrastructure Testing

All tests use **mocked agents** instead of real LLMs:

```python
def test_something(mock_agent_factory, mock_recsys_agent):
    conductor = Conductor()
    conductor.factory = mock_agent_factory  # Inject mocks

    result = conductor.run(...)

    # Verify orchestration (not LLM quality)
    assert mock_recsys_agent.execute.called
    agent_request = mock_recsys_agent.execute.call_args[0][0]
    assert agent_request.user_text == "expected query"
```

**Why mocked agents?**
- ⚡ **Fast**: <10 seconds for all 33 tests (vs 2-5 minutes with real LLMs)
- 💰 **Free**: No API costs
- ✅ **Deterministic**: Same results every time (no LLM variance)
- 🎯 **Focused**: Tests orchestration, not LLM behavior
- 🔄 **CI-friendly**: Can run on every commit

### Test Organization

```
tests/integration/chatbot/conductor/
├── conftest.py                      # Mock fixtures for all conductor tests
├── test_adapter_integrity.py        # 4 tests - Schema conversions
├── test_context_builder.py          # 8 tests - Context preparation
├── test_error_boundaries.py         # 8 tests - Error handling
├── test_multi_turn_state.py         # 5 tests - Conversation state
└── test_parameter_handling.py       # 8 tests - Parameter flow
```

**Total: 33 tests**

## Test Categories

### 1. Adapter Data Integrity (4 tests)
Tests that data survives conversions between schemas:
- TurnInput → AgentRequest (via `adapter.turn_input_to_request`)
- AgentResponse → AgentResult (via `adapter.response_to_agent_result`)
- Profile access flag propagation
- Metadata preservation (book recommendations, execution state)

**Why critical:** Silent data loss in adapters causes agent failures that look like agent bugs.

### 2. Context Builders (8 tests)
Tests `make_router_input` and `make_branch_input` logic:
- Router truncation (k_user parameter)
- Branch agent truncation (hist_turns parameter)
- Edge cases (k_user > history length, hist_turns=0)
- force_target bypass (testing utility)

**Why critical:** Nobody explicitly tests these functions. Incorrect truncation causes token overflow.

### 3. Error Boundaries (8 tests)
Tests failure handling at orchestration level:
- Agent execution failures
- Router classification failures
- Malformed inputs (empty query, whitespace, very long)
- Missing required parameters (db=None for recsys)

**Why critical:** Production agents fail. System must degrade gracefully, not crash.

### 4. Multi-Turn State Management (5 tests)
Tests conversation state across sequential Conductor calls:
- History accumulation across multiple turns
- History truncation (hist_turns parameter)
- Conversation isolation (no cross-talk between conv_ids)
- Edge cases (empty history, first turn)

**Why critical:** Production usage is always multi-turn. Component tests use static history.

### 5. Parameter Handling (8 tests)
Tests parameter propagation and edge cases:
- Cold user (num_ratings=0) vs. warm user (≥10 ratings)
- Profile access (use_profile flag)
- Optional parameters (None values)
- Metadata fields (conv_id, uid)

**Why critical:** Parameters control agent behavior and privacy. Must propagate correctly.

## Running Tests

### Prerequisites

1. **Database connection required:**
   ```bash
   export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
   ```

2. **Test users must exist:**
   - User 278859: warm user (≥10 ratings)
   - User 278857: cold user (<10 ratings)
   - User 278867: new user (0 ratings)

   Tests will attempt to use these specific users, or fall back to any users matching the criteria.

### Run All Conductor Tests

```bash
# From project root
pytest tests/integration/chatbot/conductor/ -v

# Expected: 33 passed
# Expected time: <10 seconds
```

### Run Specific Category

```bash
# Run adapter tests only
pytest tests/integration/chatbot/conductor/test_adapter_integrity.py -v

# Run error handling tests only
pytest tests/integration/chatbot/conductor/test_error_boundaries.py -v
```

### Run Single Test

```bash
pytest tests/integration/chatbot/conductor/test_adapter_integrity.py::TestAdapterDataIntegrity::test_turn_input_to_request_preserves_data -v
```

## Fixtures

### Generic Fixtures (from parent conftest.py)
- `db_session`: Database session for all tests

### Recsys Fixtures (from recsys_fixtures.py)
- `test_user_warm`: User with ≥10 ratings
- `test_user_cold`: User with <10 ratings
- `test_user_new`: User with 0 ratings
- `test_user_with_profile`: User with favorite subjects

### Mock Agent Fixtures (from conductor/conftest.py)
- `mock_recsys_agent`: Mock RecommendationAgent
- `mock_web_agent`: Mock WebAgent
- `mock_docs_agent`: Mock DocsAgent
- `mock_response_agent`: Mock ResponseAgent
- `mock_agent_factory`: Factory that returns mocked agents
- `mock_router`: Mock RouterLLM (for controlling routing)

## Expected Behavior

### Success Criteria
- **All 33 tests pass** on fresh database with test users
- **Execution time**: <10 seconds total
- **No flaky tests**: deterministic results (mocked responses)
- **No API calls**: $0.00 cost per run

### Common Failures

**"Agent execute() not called" errors:**
- Mock wasn't injected properly
- Solution: Verify `conductor.factory = mock_agent_factory` is called

**"Fixture not found" errors:**
- Fixture is in wrong conftest.py
- Solution: Check fixture is in conductor/conftest.py or parent

**Test fails but should pass:**
- Mock return value doesn't match expected schema
- Solution: Check mock configuration in conftest.py

## Key Design Patterns

### Pattern 1: Inject Mock Factory
```python
def test_something(mock_agent_factory):
    conductor = Conductor()
    conductor.factory = mock_agent_factory  # Inject mocks
    result = conductor.run(...)
```

### Pattern 2: Verify Data Flow (Not Just Calls)
```python
# Good - verify what was passed
agent_request = mock_agent.execute.call_args[0][0]
assert agent_request.user_text == "expected query"

# Bad - only verify it was called
assert mock_agent.execute.called
```

### Pattern 3: Reset Mocks Between Tests
```python
def test_multi_turn(mock_recsys_agent):
    # Turn 1
    result1 = conductor.run(...)

    # Reset mock for turn 2
    mock_recsys_agent.execute.reset_mock()

    # Turn 2
    result2 = conductor.run(...)
```

## Maintenance

### Adding New Tests
1. Choose appropriate test file based on category
2. Use existing fixtures (mock_agent_factory, test_user_warm, etc.)
3. Follow naming convention: `test_<what>_<condition>`
4. Add docstring explaining what's tested
5. Verify data flow, not just mock calls

### Updating for Code Changes
- **Conductor changes**: Update assertions in tests
- **Agent interface changes**: Update mock return values in conftest.py
- **New agent types**: Add new mock fixtures
- **Schema changes**: Update mock AgentResponse structures

## Migration from Real Agents

**Before (Phase 1):**
- Used real agents with real LLM calls
- Tests took 2-5 minutes
- Cost $0.50-2.00 per run
- Flaky due to LLM variance

**After (Phase 1.5):**
- Use mocked agents
- Tests take <10 seconds
- Cost $0.00 per run
- 100% deterministic

**What changed:**
- Added `mock_agent_factory` parameter to all tests
- Inject mocks: `conductor.factory = mock_agent_factory`
- Verify data flow instead of response content
- Assert on mock call arguments, not LLM output

**What stayed the same:**
- Test intent (what each test validates)
- Test structure and organization
- Assertions about orchestration logic
- Coverage of edge cases

## Related Testing

- **RecommendationAgent Pipeline**: `tests/integration/chatbot/agents/recommendation/`
- **Evaluation Tests**: `tests/evaluation/chatbot/` (LLM quality, not infrastructure)
- **Router Evaluation**: `tests/evaluation/chatbot/router/` (routing accuracy)

---

**For questions about these tests, refer to the implementation plan in `chatbot_testing_implementation_plan.md`**
