# app/agents/orchestrator

Entry point for all chat turns. The orchestrator receives raw request parameters from routes, decides which agent to use, and drives execution through to a final response or stream.

---

## Module layout

```
orchestrator/
├── conductor.py   # Coordinates routing → agent creation → execution
└── router.py      # LLM-based target classifier
```

---

## Request lifecycle

```
Route handler
    │  run() or run_stream()
    ▼
Conductor
    ├─ 1. RouterLLM.classify()       → Target ("recsys" | "web" | "docs" | "respond")
    ├─ 2. make_branch_input()        → TurnInput with history, profile flags, db session
    ├─ 3. AgentFactory.create_agent()→ Concrete agent instance
    ├─ 4. AgentAdapter.turn_input_to_request() → AgentRequest (domain entity)
    ├─ 5. agent.execute() / .execute_stream()
    └─ 6. AgentAdapter.response_to_agent_result() → AgentResult (legacy schema)
```

---

## Conductor

`Conductor` owns the turn-level coordination loop. All three dependencies are injected at construction time, making it straightforward to test each stage in isolation.

```python
conductor = Conductor(
    router=RouterLLM(),
    factory=AgentFactory(),
    adapter=AgentAdapter(),
)
```

### `run()` — synchronous

Returns a single `AgentResult`. Used by legacy callers and non-streaming routes.

```python
result: AgentResult = conductor.run(
    history=[{"u": "...", "a": "..."}],
    user_text="recommend me a mystery novel",
    use_profile=True,
    current_user=user,
    db=session,
    user_num_ratings=42,
)
```

### `run_stream()` — async generator

Yields `StreamChunk` objects. Used by SSE/WebSocket routes.

```python
async for chunk in conductor.run_stream(...):
    if chunk.type == "status":   ...   # progress update
    elif chunk.type == "token":  ...   # prose token
    elif chunk.type == "complete": ... # final payload
```

The conductor checks for `execute_stream` on the agent before delegating. If the agent only implements `execute()`, the conductor falls back to running it synchronously and re-emitting the text character-by-character.

### `force_target`

Any call can bypass the router with `force_target="recsys"`. Useful for testing specific agents or for routes that already know the intent (e.g., a dedicated `/recommend` endpoint).

### Error handling

Both `run()` and `run_stream()` catch all exceptions and return a safe fallback `AgentResult` / `StreamChunk(type="complete", success=False)` rather than propagating the exception to the route handler.

---

## RouterLLM

LLM-based classifier that maps a user message to one of four targets.

| Target | Intent |
|---|---|
| `recsys` | Book recommendations, "what should I read", genre/mood queries |
| `web` | Current events, external info, anything needing real-time data |
| `docs` | Help questions, "how does X work", chatbot usage queries |
| `respond` | Greetings, acknowledgements, off-topic conversation |

**Implementation notes:**

- Uses a `medium`-tier LLM with `json_mode=True`, `temperature=0`, 15 s timeout — fast and deterministic
- Only sees the system router prompt and the user text; no history, no tools
- Parses a `{"target": "...", "reason": "..."}` JSON response
- Strips markdown fences and extracts the first JSON object if the model adds prose
- If parsing fails, fires one **repair retry** asking the model to fix formatting only (not change its decision)
- Degrades to `respond` with a logged reason if both attempts fail — never raises

### Dependency injection

```python
# Testing with a mock LLM
router = RouterLLM(llm_client=my_mock_llm)
```

The `llm_client` can be any object with `.invoke()` or a plain callable — the `_chat()` method handles both.

---

## StreamChunk reference

| `type` | `content` | `data` |
|---|---|---|
| `status` | Human-readable progress string | — |
| `token` | Single prose token | — |
| `complete` | — | `target`, `success`, `book_ids`, `tool_calls`, `elapsed_ms`, `error?` |

Status chunks are emitted at these points in `run_stream()`:
1. Immediately on entry — `"Analyzing request..."`
2. After routing — `"Routing to {target} agent..."`
3. Downstream — delegated to the agent's own status hooks
