i# Conductor Integration Tests

End-to-end integration tests for the chatbot Conductor orchestration layer.

## Purpose

These tests validate integration points **not covered by component tests**:

- ✅ Adapter layer data integrity (TurnInput ↔ AgentRequest conversions)
- ✅ Multi-turn conversation state management
- ✅ Parameter propagation (user_num_ratings, use_profile, etc.)
- ✅ Error handling at orchestration level
- ✅ Context builder logic (make_router_input, make_branch_input)

**What these tests DON'T cover** (already tested elsewhere):
- ❌ Router classification accuracy → `evaluation/chatbot/router/`
- ❌ Agent tool selection logic → `evaluation/chatbot/[agent_name]/`
- ❌ Agent response quality → `evaluation/chatbot/[agent_name]/`

## Structure

```
tests/integration/chatbot/
├── conftest.py                      # Shared fixtures (test users, db)
├── test_adapter_integrity.py        # Category 1: Data conversions (4 tests)
├── test_multi_turn_state.py         # Category 2: State management (5 tests)
├── test_parameter_handling.py       # Category 3: Parameter passing (8 tests)
├── test_error_boundaries.py         # Category 4: Error handling (8 tests)
├── test_context_builders.py         # Category 5: Context building (8 tests)
└── README.md                        # This file
```

**Total: 33 integration tests**

## Test Categories

### Category 1: Adapter Data Integrity
Tests that data survives conversions between schemas:
- TurnInput → AgentRequest (via adapter.turn_input_to_request)
- AgentResponse → AgentResult (via adapter.response_to_agent_result)
- Profile access flag propagation
- Metadata preservation (book recommendations, execution state)

**Why critical:** Silent data loss in adapters causes agent failures that look like agent bugs.

### Category 2: Multi-Turn State Management
Tests conversation state across sequential Conductor calls:
- History accumulation across multiple turns
- History truncation (hist_turns parameter)
- Conversation isolation (no cross-talk between conv_ids)
- Edge cases (empty history, first turn)

**Why critical:** Production usage is always multi-turn. Component tests use static history.

### Category 3: Parameter Handling
Tests parameter propagation and edge cases:
- Cold user (num_ratings=0) vs. warm user (≥10 ratings)
- Profile access (use_profile flag)
- Optional parameters (None values)
- Metadata fields (conv_id, uid)

**Why critical:** Parameters control agent behavior and privacy. Must propagate correctly.

### Category 4: Error Boundaries
Tests failure handling at orchestration level:
- Agent execution failures
- Router classification failures
- Malformed inputs (empty query, whitespace, very long)
- Missing required parameters (db=None for recsys)

**Why critical:** Production agents fail. System must degrade gracefully, not crash.

### Category 5: Context Builders
Tests make_router_input and make_branch_input logic:
- Router truncation (k_user parameter)
- Branch agent truncation (hist_turns parameter)
- Edge cases (k_user > history length, hist_turns=0)
- force_target bypass (testing utility)

**Why critical:** Nobody explicitly tests these functions. Incorrect truncation causes token overflow.

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

### Run All Tests

```bash
# From project root
pytest tests/integration/chatbot/ -v

# Run specific category
pytest tests/integration/chatbot/test_adapter_integrity.py -v

# Run single test
pytest tests/integration/chatbot/test_adapter_integrity.py::TestAdapterDataIntegrity::test_turn_input_to_request_preserves_data -v
```

### Quick Smoke Test

```bash
# Run one test from each category (fast validation)
pytest tests/integration/chatbot/ -v -k "test_turn_input_to_request or test_history_accumulates or test_cold_user or test_empty_query or test_router_input_truncates"
```

## Expected Behavior

### Success Criteria
- **All 33 tests pass** on fresh database with test users
- **Execution time**: <2 minutes total
- **No flaky tests**: deterministic results (temperature=0 where applicable)

### Common Failures

**"User not found" errors:**
- Test users (278859, 278857, 278867) don't exist in database
- Solution: Tests will fallback to any user matching criteria, or skip if none found

**Database connection errors:**
- DATABASE_URL not set or incorrect
- Solution: Set environment variable correctly

**Agent execution timeouts:**
- LLM API slow or rate limited
- Solution: Increase timeout in agent settings or skip tests temporarily

## Design Decisions

### Why Real Database?
- Tools execute against actual data (realistic testing)
- Fast and free (no API costs for tool calls)
- Catches data schema issues

### Why Real Agents?
- Validates actual integration, not mocked behavior
- Catches version mismatches between components
- More confidence in production readiness

### Why Minimal Mocking?
- Only mock to inject failures (test error handling)
- Mock router/agent internals, not entire components
- Prefer real execution for integration validation

### Why No LLM-as-Judge?
- Testing correctness (structure, flow), not quality
- Deterministic assertions are faster and clearer
- Quality is tested separately in component evaluations

## Maintenance

### Adding New Tests
1. Choose appropriate category file
2. Add test method to corresponding class
3. Use existing fixtures (test_user_warm, test_user_cold, etc.)
4. Follow naming convention: `test_<what>_<condition>`
5. Include clear docstring explaining what's tested

### Updating for Code Changes
- **Adapter changes**: Update Category 1 tests
- **State management changes**: Update Category 2 tests
- **New parameters**: Update Category 3 tests
- **Error handling changes**: Update Category 4 tests
- **Context builder changes**: Update Category 5 tests

## Integration with CI/CD

Recommended CI configuration:

```yaml
# .github/workflows/tests.yml
integration-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run integration tests
      env:
        DATABASE_URL: ${{ secrets.DATABASE_URL }}
      run: pytest tests/integration/chatbot/ -v --timeout=120
```

## Related Testing

- **Component Tests**: `evaluation/chatbot/[agent]/` - Test individual agents
- **Router Tests**: `evaluation/chatbot/router/` - Test routing decisions
- **Unit Tests**: (if added) - Test pure functions without dependencies

## Questions?

For issues or questions about these tests:
1. Check test docstrings for specific test rationale
2. Review conftest.py for fixture behavior
3. Check component tests to understand what's NOT being tested here
