# Chatbot Integration Tests

Mock-based integration tests for chatbot infrastructure validation.

## Overview

This test suite validates chatbot infrastructure concerns using **mocked agents** instead of real LLMs. Tests are fast (<30 seconds), free ($0.00), and deterministic (100% consistent).

**Test Suite Characteristics:**
- ⚡ **Fast**: <30 seconds for all 57 tests
- 💰 **Free**: No API costs (all mocked)
- ✅ **Deterministic**: 100% consistent results
- 🔄 **CI-friendly**: Run on every commit

### Test Suite Structure

```
tests/integration/chatbot/
├── README.md                           # This file
├── conftest.py                         # Generic fixtures (db_session)
├── recsys_fixtures.py                  # Shared user fixtures
│
├── conductor/                          # Conductor orchestration (33 tests)
│   ├── README.md                       # Conductor test documentation
│   ├── conftest.py                     # Mock agent fixtures
│   ├── test_adapter_integrity.py       # 4 tests - Schema conversions
│   ├── test_context_builder.py         # 8 tests - Context preparation
│   ├── test_error_boundaries.py        # 8 tests - Error handling
│   ├── test_multi_turn_state.py        # 5 tests - Conversation state
│   └── test_parameter_handling.py      # 8 tests - Parameter flow
│
└── agents/                             # Agent-level pipelines (24 tests)
    └── recommendation/
        ├── README.md                   # Pipeline test documentation
        ├── conftest.py                 # Mock sub-agent fixtures and builders
        └── test_pipeline_integration.py # 24 tests - Pipeline data flow
```

**Total: 57 tests**

## Test Categories

### Conductor Orchestration Tests (33 tests)

Tests the Conductor's responsibility of routing user queries to appropriate agents and managing conversation flow.

**Location:** `tests/integration/chatbot/conductor/`

**What it tests:**
- Router decisions are executed correctly
- Data flows through adapter conversions (TurnInput ↔ AgentRequest ↔ AgentResult)
- Context builders truncate history correctly
- Parameters propagate through all layers
- Multi-turn conversation state is managed properly
- Errors are handled gracefully

**Mock strategy:** Mock entire agents (RecommendationAgent, WebAgent, DocsAgent, ResponseAgent)

**Documentation:** See `conductor/README.md` for detailed information

**Test breakdown:**
- Adapter integrity (4 tests): Schema conversions preserve data
- Context builders (8 tests): History truncation and preparation
- Error boundaries (8 tests): Graceful failure handling
- Multi-turn state (5 tests): Conversation persistence
- Parameter handling (8 tests): Parameter propagation

### RecommendationAgent Pipeline Tests (24 tests)

Tests the RecommendationAgent's three-stage pipeline orchestration: Planner → Retrieval → Curation.

**Location:** `tests/integration/chatbot/agents/recommendation/`

**What it tests:**
- Data flows through Planner → Retrieval → Curation stages
- Parameters reach all sub-agents correctly
- ExecutionContext is assembled properly
- Error handling works at each stage
- Complete pipeline works end-to-end

**Mock strategy:** Mock sub-agents (PlannerAgent, RetrievalAgent, CurationAgent)

**Documentation:** See `agents/recommendation/README.md` for detailed information

**Test breakdown:**
- Stage transitions (6 tests): Data flow between stages
- Parameter propagation (4 tests): Parameter passing to sub-agents
- Error handling (8 tests): Failure scenarios at each stage
- Full pipeline flow (6 tests): End-to-end execution scenarios

## Testing Architecture

### Two-Tier Testing Strategy

This suite focuses on **infrastructure testing** - validating that components work together correctly.

**Infrastructure Tests (this suite):**
- Mock agents/sub-agents to isolate orchestration logic
- Fast, free, deterministic
- Test data flow, error handling, parameter propagation
- Run on every commit in CI

**Evaluation Tests (separate suite, not included here):**
- Use real LLMs to validate output quality
- Slow, expensive, may vary
- Test LLM response quality, agent behavior, tool selection accuracy
- Run pre-deploy or on-demand

### Mock Boundaries

Tests use mocks at two different levels depending on what's being tested:

**Conductor Tests:** Mock at agent level
- Mocked components: RecommendationAgent, WebAgent, DocsAgent, ResponseAgent
- Tests: Conductor's routing and orchestration logic

**Pipeline Tests:** Mock at sub-agent level
- Mocked components: PlannerAgent, RetrievalAgent, CurationAgent
- Tests: RecommendationAgent's pipeline orchestration logic

**Rationale:** Clean boundaries at integration points, stable interfaces, easier maintenance

### Fixture-Based Mock Injection

Tests use pytest fixtures that return pre-configured mocked components:

```python
# Conductor tests use mock_agent_factory
@pytest.fixture
def mock_agent_factory(mock_recsys_agent, mock_web_agent, ...):
    """Returns factory that creates mocked agents."""
    factory = Mock()
    factory.create_agent = lambda agent_type, **kwargs: {
        "recsys": mock_recsys_agent,
        "web": mock_web_agent,
    }[agent_type]
    return factory

# Pipeline tests use builders
mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
mock_curation = mock_curation_builder.returns_success_with_books(5).build()
```

**Benefits:** Consistent mock behavior, easy customization per test, minimal boilerplate

## Running Tests

### Prerequisites

1. **Database connection:**
   ```bash
   export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
   ```

2. **Test users must exist:**
   - User 278859: warm user (≥10 ratings)
   - User 278857: cold user (<10 ratings)
   - User 278867: new user (0 ratings)

### Run All Integration Tests

```bash
# From project root
pytest tests/integration/chatbot/ -v

# Expected: 57 passed
# Expected time: <30 seconds
```

### Run Tests by Component

```bash
# Conductor tests (33 tests)
pytest tests/integration/chatbot/conductor/ -v

# Pipeline tests (24 tests)
pytest tests/integration/chatbot/agents/recommendation/ -v
```

### Run Specific Category

```bash
# Adapter integrity tests (4 tests)
pytest tests/integration/chatbot/conductor/test_adapter_integrity.py -v

# Stage transitions tests (6 tests)
pytest tests/integration/chatbot/agents/recommendation/test_pipeline_integration.py::TestStageTransitions -v
```

## CI/CD Integration

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
      run: |
        pytest tests/integration/chatbot/ -v --timeout=60
        # Should complete in <30 seconds, 60s is generous buffer
```

**Key points:**
- ✅ Fast enough to run on every commit
- ✅ Deterministic results (no flaky tests)
- ✅ No API costs (all mocked)
- ✅ Clear failure messages (data flow validation)

## What These Tests DON'T Cover

These are **infrastructure tests**, not **quality tests**.

**What they validate:**
- ✅ "Does the right component get called with the right data?"
- ✅ "Does data flow correctly through the system?"
- ✅ "Are errors handled gracefully?"
- ✅ "Do parameters propagate correctly?"

**What they DO NOT validate:**
- ❌ "Is the LLM response helpful?" → Evaluation tests
- ❌ "Did the router pick the right agent?" → Router evaluation tests
- ❌ "Are the book recommendations relevant?" → Quality evaluation tests
- ❌ "Is the tool selection optimal?" → Component evaluation tests

**Separation of concerns:**
- **Infrastructure** (this suite) → Fast, deterministic, run frequently, test plumbing
- **Quality** (separate suite) → Slow, may vary, run infrequently, test outcomes

## Fixture Hierarchy

```
tests/integration/chatbot/
│
├── conftest.py                         # Level 1: Generic
│   └── db_session                      # Any test can use
│
├── recsys_fixtures.py                  # Level 2: Domain-specific
│   ├── test_user_warm                  # Recsys tests can use
│   ├── test_user_cold
│   ├── test_user_new
│   └── test_user_with_profile
│
├── conductor/
│   └── conftest.py                     # Level 3: Test-specific
│       ├── (imports recsys_fixtures)
│       ├── mock_recsys_agent           # Conductor tests only
│       ├── mock_web_agent
│       ├── mock_docs_agent
│       ├── mock_response_agent
│       └── mock_agent_factory
│
└── agents/recommendation/
    └── conftest.py                     # Level 3: Test-specific
        ├── (imports recsys_fixtures)
        ├── mock_planner_builder        # Pipeline tests only
        ├── mock_retrieval_builder
        ├── mock_curation_builder
        ├── candidate_factory
        └── strategy_factory
```

**Principle:** Fixtures flow down (child can use parent), not sideways (conductor ≠ recommendation).

## Maintenance

### Adding New Tests

**For Conductor tests:**
1. Identify appropriate category file in `conductor/`
2. Add test method to corresponding test class
3. Use existing fixtures (`mock_agent_factory`, `test_user_warm`, etc.)
4. Follow naming convention: `test_<what>_<condition>`
5. Include clear docstring explaining what's tested

**For Pipeline tests:**
1. Identify appropriate test class in `test_pipeline_integration.py`
2. Use builders (`mock_planner_builder`, `mock_retrieval_builder`, etc.)
3. Verify data flow through the pipeline, not just that mocks were called
4. Include docstring explaining what's tested

### Updating for Code Changes

**Agent interface changes:**
- Update mock return values in `conductor/conftest.py`
- Verify AgentResponse schema matches current implementation

**Sub-agent interface changes:**
- Update mock return values in `recommendation/conftest.py`
- Update PlannerStrategy, RetrievalInput, ExecutionContext schemas

**Schema changes:**
- Update mock data structures to match new fields
- Update test assertions to verify new fields
- Update factory methods to generate correct test data

**New agents:**
- Add mock fixture to `conductor/conftest.py`
- Add mapping in `mock_agent_factory`
- Add corresponding test coverage

## Common Patterns

### Pattern 1: Inject Mocks
```python
def test_something(mock_agent_factory):
    conductor = Conductor()
    conductor.factory = mock_agent_factory  # Inject mocks
    result = conductor.run(...)
```

### Pattern 2: Verify Data Flow
```python
# Good - verify what was passed
agent_request = mock_agent.execute.call_args[0][0]
assert agent_request.user_text == "expected query"

# Bad - only verify it was called
assert mock_agent.execute.called
```

### Pattern 3: Use Builders
```python
# Good - fluent API
mock_planner = mock_planner_builder.returns_warm_user_strategy().build()

# Bad - manual configuration
mock_planner = Mock()
mock_planner.execute.return_value = ...
```

## Troubleshooting

### Common Issues

**"User not found" errors:**
- Test users don't exist in database
- Solution: Tests will fallback to any matching user, or skip if none found

**"Mock not called" errors:**
- Mock wasn't injected properly
- Solution: Verify dependency injection (`conductor.factory = mock_agent_factory`)

**"Database connection failed":**
- DATABASE_URL not set or incorrect
- Solution: Set environment variable correctly

**Test fails unexpectedly:**
- Mock return value doesn't match expected schema
- Solution: Check mock configuration in conftest.py

## Related Documentation

- **Conductor Tests**: See `conductor/README.md` for detailed Conductor test documentation
- **Pipeline Tests**: See `agents/recommendation/README.md` for detailed pipeline test documentation
- **Evaluation Tests**: (Separate suite) Tests for LLM quality and agent behavior

---

**Questions?**

For detailed information about specific test components:
- Conductor orchestration: See `conductor/README.md`
- Pipeline orchestration: See `agents/recommendation/README.md`
- Fixture definitions: See `conftest.py` files at each level
