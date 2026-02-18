# app/agents/infrastructure/recsys

Multi-agent pipeline for personalized book recommendations.
Designed for **streaming-first** execution: Planner and Retrieval run to completion, then Curation streams token-by-token to the client.

---

## Architecture

```
User Query
    │
    ▼
RecommendationAgent (Orchestrator)
    │
    ├── Stage 1 ─ PlannerAgent          (single async LLM call, no streaming)
    │       └── Determines retrieval strategy + optional profile pre-fetch
    │
    ├── Stage 2 ─ RetrievalAgent        (LangGraph ReAct loop, no streaming)
    │       └── Executes strategy, gathers 60–120 candidate books
    │
    └── Stage 3 ─ CurationAgent         (LangGraph, streaming)
            └── Ranks, filters, writes prose with inline citations → yields tokens
```

The primary entry point is `RecommendationAgent.execute_stream()`, which yields `StreamChunk` objects (status updates, prose tokens, and a final completion payload).  A synchronous `execute()` method is also available for backward-compatible callers.

---

## Module layout

```
recsys/
├── __init__.py            # Public exports
├── orchestrator.py        # RecommendationAgent – wires all three stages
├── planner_agent.py       # Stage 1 – query analysis and strategy selection
├── retrieval_agent.py     # Stage 2 – candidate generation via retrieval tools
└── curation_agent.py      # Stage 3 – ranking, prose, inline citations (streaming)
```

---

## Stages in detail

### Stage 1 – PlannerAgent

Makes **one** async LLM call (`medium` tier, JSON mode, `temperature=0`).

**Inputs**
- User query
- Available retrieval tools (ALS included only if `user_num_ratings >= warm_threshold`)
- User profile data (fetched directly from DB when `allow_profile=True`)

**Outputs** – `PlannerStrategy`
- `recommended_tools` – ordered list of tools to try first
- `fallback_tools` – alternatives if primary tools underperform
- `reasoning` – natural-language explanation of the strategy
- `profile_data` – attached profile dict (favourite subjects with IDs + recent interactions)

Profile data is fetched **before** the LLM call and injected into the prompt, so the LLM sees subject IDs it can pass directly to `subject_hybrid_pool`.

---

### Stage 2 – RetrievalAgent

Runs a guided LangGraph ReAct loop (`large` tier, up to 6 iterations, 60 s timeout).

**Available tools**

| Tool | When available |
|---|---|
| `als_recs` | Warm users (≥ `warm_threshold` ratings) |
| `book_semantic_search` | Always |
| `subject_hybrid_pool` | Always |
| `subject_id_search` | Always |
| `popular_books` | Always (cold-user fallback) |

Profile-context tools (`user_profile`, `recent_interactions`) are intentionally **not** registered here — PlannerAgent handles all profile access.

**Stopping criteria** (evaluated by the LLM)
- ≥ 60 candidates → ideal range
- ≥ 120 candidates → stop immediately
- All recommended + fallback tools exhausted
- Iteration limit reached

The agent returns candidates even on partial failure (timeout / tool error), so Stage 3 always receives _something_ to work with.

**Outputs** – `RetrievalOutput`
- `candidates` – list of `BookRecommendation` dicts
- `execution_context` – planner reasoning, tools used, profile data forwarded to Curation
- `tool_execution_summaries` – lightweight audit trail

---

### Stage 3 – CurationAgent  *(streaming)*

Single LangGraph call (`large` tier, 1 iteration, 30 s timeout, **no tools**).

Receives the full candidate list plus execution context and writes **natural-language prose** with markdown-style inline citations:

```
I recommend [The Name of the Rose](702) for its masterful mystery atmosphere.
```

The citation format `[Title](item_idx)` is parsed on completion to extract an ordered list of cited book IDs. Citations are validated against the candidate set and any hallucinated IDs are flagged in logs.

**`execute_stream()`** yields:

| Chunk type | Content |
|---|---|
| `status` | `"Curating personalized recommendations..."` |
| `token` | Individual prose tokens as they arrive |
| `complete` | `{ book_ids, tools_used, success, elapsed_ms, … }` |

---

## Streaming flow

```python
async for chunk in recommendation_agent.execute_stream(request):
    if chunk.type == "status":
        # Show spinner / progress message
    elif chunk.type == "token":
        # Append token to the response stream
    elif chunk.type == "complete":
        book_ids = chunk.data["book_ids"]   # ordered by citation appearance
        tools_used = chunk.data["tools_used"]
```

Status chunks are emitted at the start of each stage (Planning, Retrieval, Curation) so the UI can show incremental progress before prose begins.

---

## Instantiation

```python
agent = RecommendationAgent(
    current_user=user,          # SQLAlchemy user object
    db=db_session,              # SQLAlchemy session
    user_num_ratings=42,        # Determines ALS availability
    warm_threshold=10,          # Minimum ratings for ALS
    allow_profile=True,         # User consented to profile access
)
```

Sub-agents can be injected for testing:

```python
agent = RecommendationAgent(
    planner_agent=mock_planner,
    retrieval_agent=mock_retrieval,
    curation_agent=mock_curation,
)
```

---

## User warm/cold logic

| Condition | ALS tool | Profile fetch |
|---|---|---|
| `num_ratings >= warm_threshold` AND `allow_profile=True` | ✅ | ✅ |
| `num_ratings >= warm_threshold` | ✅ | ❌ |
| `num_ratings < warm_threshold` | ❌ | ❌ (falls back to `popular_books`) |

---

## Error handling

Each stage has an independent fallback so a failure does not cascade:

- **Planning failure** → polite error message, no books returned
- **Retrieval failure** → polite error message (partial candidates are still forwarded if available)
- **No candidates** → context-aware message (hints at pre-2004 catalog limit if query mentions recent dates)
- **Curation failure** → top 10 raw candidates returned without prose

---

## Logging

All structured logging uses `append_chatbot_log()` with verbosity controlled by the `LOG_MODE` environment variable.  Component-level filtering is available via `should_log_component()`.  Set `DEBUG=true` to enable citation-level diagnostics in the Curation stage.
