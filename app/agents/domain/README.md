# app/agents/domain

Framework-agnostic business logic layer. All core types, contracts, and services live here — no LangGraph, no LangChain, no FastAPI imports.

The goal of this layer is to let the infrastructure layer (LangGraph agents, tool registries) and the application layer (routes, conductor) depend on stable abstractions rather than on each other.

---

## Module layout

```
domain/
├── __init__.py         # Curated public exports
├── entities.py         # Core mutable and immutable business objects
├── interfaces.py       # Abstract contracts (ABC + Protocol)
├── services.py         # StandardResultProcessor — book extraction logic
├── recsys_schemas.py   # Data structures for the three-stage recsys pipeline
├── value_objects.py    # Immutable configuration primitives
└── parsers.py          # Inline book-tag parser (backend validation only)
```

---

## Entities

Defined in `entities.py`. These are the shared language of the whole system.

### `AgentConfiguration` *(frozen dataclass)*

Immutable per-agent settings. Set once at `__init__`, never mutated.

| Field | Purpose |
|---|---|
| `policy_name` | Prompt file to load (e.g. `"web.system.md"`) |
| `capabilities` | `frozenset[AgentCapability]` — what the agent can do |
| `allowed_tools` | `frozenset[str]` — tools the agent may call |
| `llm_tier` | `"small"` / `"medium"` / `"large"` |
| `timeout_seconds` | Hard execution deadline |
| `max_iterations` | ReAct loop cap |

### `AgentRequest`

Carries everything an agent needs: `user_text`, `conversation_history`, and an `ExecutionContext` with session data and user preferences. Immutable-style — use `.with_context(**kwargs)` to derive a modified copy.

### `AgentResponse`

Carries everything a caller needs back: `text`, `target_category`, `success`, `book_recommendations`, `citations`, and the optional `execution_state` for diagnostics. Computed properties `.execution_time_ms` and `.tool_calls_count` derive from the attached state.

### `BookRecommendation`

The canonical book object used throughout the pipeline. Core identity fields (`item_idx`, `title`, `author`, `year`) plus optional enrichment (`subjects`, `tones`, `vibe`, `genre`). `.to_curation_dict()` produces a token-efficient dict for the curation agent context window — empty fields are omitted automatically.

### `AgentExecutionState`

Mutable, append-only record of what happened during a run: tool executions, reasoning steps, intermediate outputs. `.mark_completed()` and `.mark_failed()` set the terminal status and record `end_time`. `.execution_time_ms` is derived from `start_time`/`end_time`.

### `ToolExecution`

Single tool call record: name, arguments, result, optional error, and timing. `.succeeded` is `True` when `error is None`.

---

## Interfaces

Defined in `interfaces.py`. Three contracts, two styles:

**`BaseAgent` (ABC)** — for classes. Enforces `execute(request) → AgentResponse`. All concrete agents inherit from this via `BaseLangGraphAgent`.

**`Agent` (Protocol, `@runtime_checkable`)** — for structural typing and duck typing in tests. An object satisfies `Agent` if it has `execute()` and `configuration` without inheriting anything.

**`ToolProvider` (Protocol)** — `get_available_tools()`, `execute_tool()`, `is_tool_allowed()`. Used by `ToolRegistry`.

**`ResultProcessor` (Protocol)** — `extract_book_recommendations()`, `format_response_text()`, `extract_citations()`. Used by `StandardResultProcessor`.

---

## StandardResultProcessor

Defined in `services.py`. Transforms raw `AgentExecutionState` into structured `BookRecommendation` lists.

### Extraction priority order

```
1. book_objects  (full metadata dicts in state.intermediate_outputs)   ← preferred
2. return_book_ids tool result  (most recent, parsed from JSON)
3. return_book_ids tool arguments  (fallback if result unparseable)
4. Other tool outputs  (regex scan for item_idx patterns)
5. Reasoning steps  (least reliable, last resort)
```

Each strategy is tried in order; the first one that returns any IDs wins. All failures are silently swallowed and the next strategy is attempted — the processor never raises.

`_build_recommendations_from_objects()` is the preferred path (used by the recsys pipeline). It maps raw tool result dicts to fully populated `BookRecommendation` objects including enrichment metadata.

---

## Recsys schemas

Defined in `recsys_schemas.py`. Typed data contracts for the three-stage recommendation pipeline. Each stage has an explicit input and output type so the orchestrator can be read top-to-bottom without needing to know internal stage details.

```
PlannerInput  →  PlannerAgent  →  PlannerStrategy
                                        │
                              RetrievalInput (wraps strategy)
                                        │
                              RetrievalAgent  →  RetrievalOutput
                                                      │
                                            ExecutionContext (forwarded to Curation)
```

| Schema | Role |
|---|---|
| `PlannerInput` | Query + user warmth flags + available tool list |
| `PlannerStrategy` | Recommended tools, fallback tools, reasoning, optional profile data |
| `RetrievalInput` | Query + strategy (passes strategy through untouched) |
| `RetrievalOutput` | 60–120 candidate dicts + `ExecutionContext` + optional `ToolExecutionSummary` list |
| `ExecutionContext` | Planner reasoning + tools used + profile data — forwarded to Curation |
| `ToolExecutionSummary` | Lightweight audit record (no result payload) — used in evals, not production |
| `CurationInput` | Query + candidates + execution context (defined for completeness; Curation receives these as kwargs) |

---

## Value objects

Defined in `value_objects.py`. All frozen dataclasses — safe to use as dict keys or in sets.

| Class | Purpose |
|---|---|
| `AgentType` | Enum of agent categories |
| `ToolPermissions` | Immutable tool allow-list with access flags; `.with_additional_tools()` returns a new instance |
| `ModelConfiguration` | LLM tier + temperature + timeout; `.for_creativity()` / `.for_analysis()` derive preset configs |
| `ExecutionLimits` | Hard caps on iterations, time, tool calls, and reasoning steps; `.is_within_limits()` checks all four at once |

---

## InlineReferenceParser

Defined in `parsers.py`. Backend-only validator for the `<book id="X">Title</book>` citation format used by the curation agent.

**Not used in the response path** — the frontend parses HTML tags directly from the response text. This parser is used after the fact for quality control.

| Method | Purpose |
|---|---|
| `extract_book_tags(text)` | Returns `List[BookTagMatch]` with position info |
| `validate_references(text, book_recommendations)` | Returns `(errors, warnings)` — errors for hallucinated IDs, warnings for duplicates or over-tagging |
| `strip_book_tags(text)` | Removes tags, leaves inner text — for plain text rendering |
| `get_inline_book_ids(text)` | Returns IDs in order of appearance — for logging/analytics |

Validation rules:
- Any cited `id` not present in `book_recommendations` → **error** (hallucinated ID)
- Same book tagged more than once → **warning**
- More than 12 unique books tagged → **warning** (readability guideline)
