# Streaming Infrastructure Implementation

## Overview

The streaming infrastructure has been successfully implemented across three layers:
1. **Schema Layer**: StreamChunk model for streaming data
2. **Orchestration Layer**: Conductor.run_stream() for async streaming
3. **HTTP Layer**: /chat/stream endpoint for Server-Sent Events

## Files Modified

### 1. app/agents/schemas.py
**Added:**
- `StreamChunk` Pydantic model with three types:
  - `status`: Progress updates (e.g., "Analyzing request...")
  - `token`: Individual tokens of the final response
  - `complete`: Final result with metadata

### 2. app/agents/orchestrator/conductor.py
**Added:**
- Import: `AsyncGenerator` from typing
- Import: `StreamChunk` from schemas
- Method: `async run_stream()` - Async streaming version of run()

**Key Features:**
- Yields status updates at key orchestration points
- Detects if agents support streaming (via `execute_stream` method)
- Falls back to synchronous execution for non-streaming agents
- Handles errors gracefully by yielding error completion chunks

### 3. routes/chat.py (or chat.py)
**Added:**
- Import: `StreamingResponse` from fastapi.responses
- Import: `json` module
- Endpoint: `POST /chat/stream` - Streaming version of /chat/agent

**Key Features:**
- Server-Sent Events (SSE) format
- Rate limiting and authentication (same as /chat/agent)
- History management (saves after streaming completes)
- Proper SSE headers (Cache-Control, X-Accel-Buffering)

### 4. test_streaming.py (NEW)
**Purpose:**
Demonstrates and validates the streaming infrastructure with a mock agent.

**Components:**
- `MockStreamingAgent`: Emits test chunks to validate pipeline
- `test_conductor_streaming()`: Tests Conductor streaming
- `test_sse_format()`: Validates SSE message format

## Architecture

### Flow Diagram

```
Client Request
    |
    v
POST /chat/stream
    |
    v
Conductor.run_stream()
    |
    +-- Yield: StreamChunk(type="status", content="Analyzing request...")
    |
    +-- Route to target agent
    |
    +-- Yield: StreamChunk(type="status", content="Routing to X agent...")
    |
    +-- Create agent
    |
    +-- Check if agent.execute_stream() exists
    |
    +-- YES: Stream from agent
    |   |
    |   +-- Agent yields status updates
    |   +-- Agent yields token chunks
    |   +-- Agent yields completion
    |
    +-- NO: Fallback to sync execution
        |
        +-- Execute synchronously
        +-- Yield tokens character-by-character
        +-- Yield completion
```

### StreamChunk Structure

```python
# Status update
{
    "type": "status",
    "content": "Searching documentation...",
    "data": null
}

# Token (part of response text)
{
    "type": "token",
    "content": "word ",
    "data": null
}

# Completion (final result)
{
    "type": "complete",
    "content": null,
    "data": {
        "target": "docs",
        "success": true,
        "book_ids": [123, 456],
        "tool_calls": [...],
        "citations": [...],
        "policy_version": "conductor.mas.v1",
        "elapsed_ms": 1234
    }
}
```

### SSE Format

Server-Sent Events are sent as:
```
data: {"type": "status", "content": "Processing..."}\n\n
data: {"type": "token", "content": "Hello "}\n\n
data: {"type": "token", "content": "world"}\n\n
data: {"type": "complete", "data": {...}}\n\n
```

## How to Test

### 1. Run the Test Script

```bash
cd /path/to/project
python test_streaming.py
```

This validates:
- Conductor streaming pipeline
- StreamChunk format
- SSE message structure

### 2. Test with curl

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "What are some good mystery books?",
    "use_profile": false
  }' \
  --no-buffer
```

You should see SSE events streaming in real-time.

### 3. Test with Frontend

JavaScript example:
```javascript
const eventSource = new EventSource('/chat/stream', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        user_text: 'Recommend some sci-fi books',
        use_profile: false
    })
});

eventSource.onmessage = (event) => {
    const chunk = JSON.parse(event.data);

    if (chunk.type === 'status') {
        console.log('Status:', chunk.content);
    } else if (chunk.type === 'token') {
        // Append to response display
        responseDiv.textContent += chunk.content;
    } else if (chunk.type === 'complete') {
        console.log('Complete:', chunk.data);
        eventSource.close();
    }
};
```

## Backward Compatibility

The streaming infrastructure is **fully backward compatible**:

1. **Original endpoint unchanged**: `/chat/agent` still works synchronously
2. **New streaming endpoint**: `/chat/stream` is opt-in
3. **Agent fallback**: Non-streaming agents still work via character-by-character fallback
4. **Same authentication/rate limiting**: Both endpoints use identical security

## Next Steps

### Phase 2: Implement Agent Streaming

Now that infrastructure is ready, implement `execute_stream()` in agents:

#### Priority Order:
1. **DocsAgent** (simplest - 2 tools, no complex logic)
2. **WebAgent** (moderate - external APIs)
3. **ResponseAgent** (trivial - direct LLM streaming)
4. **RecommendationAgent** (complex - 3-stage pipeline)

#### Example Implementation (DocsAgent):

```python
async def execute_stream(self, request: AgentRequest):
    # Status: Starting
    yield StreamChunk(type='status', content='Searching documentation...')

    # Execute agent logic
    # ... (use prebuilt agent or custom graph)

    # Status: Found results
    yield StreamChunk(type='status', content='Generating response...')

    # Stream response tokens
    for token in response_tokens:
        yield StreamChunk(type='token', content=token)

    # Completion
    yield StreamChunk(type='complete', data={...})
```

### Migration Path

1. Start with DocsAgent streaming implementation
2. Test thoroughly with `/chat/stream` endpoint
3. Once validated, implement WebAgent streaming
4. Then ResponseAgent (simplest)
5. Finally, RecommendationAgent (most complex)
6. After all agents support streaming, consider deprecating sync endpoint

## Error Handling

The infrastructure handles errors at multiple levels:

1. **Conductor Level**: Catches exceptions and yields error completion chunk
2. **Endpoint Level**: Wraps in try/catch and returns error SSE
3. **Agent Level**: Agents can yield error status updates before failing

Example error flow:
```python
try:
    async for chunk in agent.execute_stream(request):
        yield chunk
except Exception as e:
    yield StreamChunk(
        type="complete",
        data={
            "success": False,
            "error": str(e)
        }
    )
```

## Performance Considerations

1. **Buffering**: SSE headers disable buffering for immediate delivery
2. **Memory**: Tokens are streamed immediately, not accumulated
3. **History**: Only saved after completion (not per-token)
4. **Rate Limiting**: Applied once at request start, not per chunk

## Security

All existing security measures maintained:
- Authentication checks
- Rate limiting (per-user and system-wide)
- Input validation
- Session management
- CORS headers

## Known Limitations

1. **No cancellation**: Currently no way to cancel mid-stream
2. **No reconnection**: Client must handle reconnection on disconnect
3. **Single response**: One request = one stream (no bidirectional)
4. **Agent compatibility**: Agents must implement `execute_stream()` or use fallback

## Future Enhancements

1. Add cancellation support (via abort signals)
2. Implement reconnection with event IDs
3. Add progress percentages to status updates
4. Support partial results for long-running operations
5. Add telemetry for streaming performance
