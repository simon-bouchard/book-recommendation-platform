# app/agents/infrastructure

Agent infrastructure layer — concrete agent implementations built on LangGraph's prebuilt ReAct engine.
This module owns everything between the routing decision (made upstream by the Conductor) and the final `AgentResponse` / `StreamChunk` stream returned to the caller.

---

## Architecture

```
Conductor / Router
        │
        ▼
  AgentFactory.create_agent(target, ...)
        │
        ├── "web"      → WebAgent
        ├── "docs"     → DocsAgent
        ├── "respond"  → ResponseAgent
        └── "recsys"   → RecommendationAgent  (see recsys/README.md)
                │
                └─ BaseLangGraphAgent (shared base)
                        ├── LangGraph create_react_agent
                        ├── ToolRegistry / ToolExecutor
                        └── StandardResultProcessor
```

---

## Module layout

```
infrastructure/
├── __init__.py                  # Public exports
├── base_langgraph_agent.py      # Shared base class for all agents
├── agent_factory.py             # Creates agents by routing target
├── agent_adapter.py             # Converts legacy schemas ↔ domain entities
├── tool_executor.py             # Executes native tools with type coercion
├── web_agent.py                 # Web search + synthesis
├── docs_agent.py                # Internal documentation search
├── response_agent.py            # Conversational fallback (no tools)
└── recsys/                      # Three-stage recommendation pipeline
    └── (see recsys/README.md)
```

---

## BaseLangGraphAgent

`base_langgraph_agent.py` is the foundation every concrete agent inherits from.  It wraps LangGraph's `create_react_agent` and provides a consistent interface so subclasses only define _what_ they do, not _how_ the loop runs.

### What it provides

| Feature | Detail |
|---|---|
| ReAct loop | LangGraph `create_react_agent` — no custom graph needed |
| Native function calling | Tools bound via `llm.bind_tools()`, no JSON parsing |
| `execute()` | Async, returns `AgentResponse` |
| `execute_stream()` | Async generator, yields `StreamChunk` objects |
| Message building | `_build_messages()` with structured injection points |
| Status hooks | Per-agent and per-tool status messages for streaming UI |
| Timeout | Thread-based timeout guard per `configuration.timeout_seconds` |
| Result processing | `StandardResultProcessor` shared across all agents |

### Message assembly order

```
SystemMessage  ← _get_system_prompt()
[context msgs] ← _add_context_messages(**context)   # override per agent
[history]      ← last 3 conversation turns
HumanMessage   ← current user query
```

### Subclass contract

Three abstract methods must be implemented:

```python
def _get_system_prompt(self) -> str: ...
def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry: ...
def _get_target_category(self) -> str: ...
```

Optional overrides for finer control:

```python
def _add_context_messages(self, **context) -> List[BaseMessage]: ...  # inject context
def _get_start_status(self) -> str: ...                               # initial status
def _get_tool_start_status(self, tool_name, args) -> str: ...         # per-tool status
def _get_tool_complete_status(self) -> str: ...                       # after tool runs
```

### StreamChunk types

| Type | When emitted | Key fields |
|---|---|---|
| `status` | Start of execution, before/after each tool | `content` |
| `token` | Each streaming token from the LLM | `content` |
| `complete` | Execution finished (success or error) | `target`, `text`, `success`, `tool_calls`, `elapsed_ms` |

---

## Concrete agents

### WebAgent

LangGraph ReAct agent for external information retrieval.

| Property | Value |
|---|---|
| LLM tier | `large` |
| Timeout | 60 s |
| Max iterations | 10 |
| Tools | `web_search`, `web_fetch` |
| Streaming | ✅ (via base class) |

Injects the current date as a context message so the agent can reason about when to use web search versus training data.  System prompt is composed from `persona.system.md` + `web.system.md`.

---

### DocsAgent

ReAct agent for internal help documentation lookup.

| Property | Value |
|---|---|
| LLM tier | `large` |
| Timeout | 30 s |
| Max iterations | 5 |
| Tools | `help_read` |
| Streaming | ✅ (via base class) |

Loads the docs manifest at init and appends it to the system prompt (up to 10 items) so the LLM knows what documents are available before deciding which to read.  Manifest load failure is silently swallowed — it does not break agent creation.

System prompt is composed from `persona.system.md` + `docs.system.md` + manifest.

---

### ResponseAgent

No-tools conversational fallback.

| Property | Value |
|---|---|
| LLM tier | `medium` |
| Timeout | 20 s |
| Max iterations | 1 |
| Tools | none |
| Streaming | ✅ (via base class) |

Single-pass execution.  Used for greetings, acknowledgements, and anything the router classifies as `"respond"`.  System prompt is `persona.system.md` plus a brief instruction to stay concise.

---

### RecommendationAgent (recsys)

Three-stage pipeline — see [`recsys/README.md`](recsys/README.md) for full documentation.

---

## AgentFactory

Creates agents by routing target with optional provider injection for testing.

```python
factory = AgentFactory()

agent = factory.create_agent(
    target="recsys",
    current_user=user,
    db=db_session,
    user_num_ratings=42,
    use_profile=True,
)
```

All four targets (`recsys`, `web`, `docs`, `respond`) map to provider callables that default to the real agents but can be replaced in tests:

```python
factory = AgentFactory(
    recsys_provider=lambda **kw: MockRecommendationAgent(),
    web_provider=lambda **kw: MockWebAgent(),
)
```

The factory applies the warm-user threshold check (`num_ratings >= 10`) for the recsys target before forwarding the flag to `RecommendationAgent`.

---

## AgentAdapter

Translates between the legacy request/response schemas used by routes and the domain entities used by agents.

| Method | Direction |
|---|---|
| `turn_input_to_request(turn_input)` | `TurnInput` → `AgentRequest` |
| `response_to_agent_result(response)` | `AgentResponse` → `AgentResult` |
| `book_recommendations_to_book_out(recs)` | `List[BookRecommendation]` → `List[BookOut]` |

This keeps domain entities clean and isolates the legacy schema contract to a single boundary class.

---

## ToolExecutor

Executes `ToolDefinition` instances and returns typed `ToolExecution` domain entities.

Key behaviours:

- **Type coercion** — handles `Optional[X]`, `list[int]`, `dict[str, Any]`, string-to-bool, JSON-string-to-list/dict
- **Parameter validation** — raises `ValueError` with a clear message for missing required params
- **Safe fallback** — any exception is caught and returned as a failed `ToolExecution` (never raises to the caller)

Helper methods used by the result processor:

| Method | Purpose |
|---|---|
| `extract_book_ids_from_result(result)` | Pull `item_idx` values from any result shape |
| `extract_books_from_result(result)` | Pull full book dicts (with metadata) from any result shape |
| `is_book_recommendation_tool(name)` | Check if a tool returns book candidates |
| `get_tool_info(name)` | Introspect a tool's signature and metadata |

---

## Adding a new agent

1. Create `my_agent.py` inheriting from `BaseLangGraphAgent`
2. Implement the three abstract methods
3. Add a new target branch in `AgentFactory.create_agent()`
4. Export from `__init__.py`

Minimal skeleton:

```python
class MyAgent(BaseLangGraphAgent):
    def __init__(self):
        super().__init__(AgentConfiguration(
            policy_name="my.system.md",
            capabilities=frozenset([AgentCapability.CONVERSATIONAL]),
            allowed_tools=frozenset(["my_tool"]),
            llm_tier="medium",
            timeout_seconds=30,
            max_iterations=5,
        ))

    def _get_system_prompt(self) -> str:
        return read_prompt("my.system.md")

    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        return ToolRegistry(my_tools=True, gates=InternalToolGates())

    def _get_target_category(self) -> str:
        return "my_target"
```
