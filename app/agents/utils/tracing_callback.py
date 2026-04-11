# app/agents/utils/tracing_callback.py
"""
OpenTelemetry tracing callback for LangGraph tool executions.

Opens a child span for each tool call so that inner httpx calls (model server
requests) are correctly nested under the tool span in Jaeger rather than
appearing as siblings of the retrieval stage span.

Used alongside RetrievalLoggingCallback in RetrievalAgent.execute().
"""

import json
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)


class ToolTracingCallback(BaseCallbackHandler):
    """
    LangChain callback that wraps each tool execution in an OTEL span.

    Uses a stack so that nested tool calls are handled safely, though the
    ReAct loop executes tools sequentially in practice.

    Span naming: chat.tool.<tool_name>
    Attributes set on each span:
        tool.name           tool function name
        tool.arg.*          scalar/list-length arguments (truncated to 200 chars)
        tool.result_count   number of books returned (for retrieval tools)
        tool.succeeded      True on success, False on error
    """

    def __init__(self) -> None:
        super().__init__()
        self._span_stack: List[Any] = []

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")

        # Explicitly capture the current parent so the span is correctly parented
        # even if LangGraph switches asyncio task contexts between callbacks.
        # We do NOT attach/detach here — that causes ValueError when LangGraph
        # runs tool invocations in a copied context (different ContextVar slot).
        span = tracer.start_span(
            f"chat.tool.{tool_name}",
            context=otel_context.get_current(),
        )
        span.set_attribute("tool.name", tool_name)

        try:
            args = json.loads(input_str)
            if isinstance(args, dict):
                for k, v in args.items():
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(f"tool.arg.{k}", str(v)[:200])
                    elif isinstance(v, list):
                        span.set_attribute(f"tool.arg.{k}.count", len(v))
        except (json.JSONDecodeError, Exception):
            pass

        self._span_stack.append(span)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        if not self._span_stack:
            return

        span = self._span_stack.pop()
        output_str = output.content if hasattr(output, "content") else str(output)
        span.set_attribute("tool.result_count", self._count_results(output_str))
        span.set_attribute("tool.succeeded", True)
        span.end()

    def on_tool_error(self, error: Any, **kwargs: Any) -> None:
        if not self._span_stack:
            return

        span = self._span_stack.pop()
        exc = error if isinstance(error, Exception) else Exception(str(error))
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR))
        span.set_attribute("tool.succeeded", False)
        span.end()

    # -------------------------------------------------------------------------

    def _count_results(self, output: str) -> int:
        if not output or not isinstance(output, str):
            return 0
        try:
            data = json.loads(output)
            if isinstance(data, list):
                return len([x for x in data if isinstance(x, dict) and "error" not in x])
            if isinstance(data, dict):
                for key in ("books", "results", "items", "candidates"):
                    if key in data and isinstance(data[key], list):
                        return len(data[key])
        except Exception:
            pass
        return 0
