# app/agents/infrastructure/base_langgraph_agent.py
"""
Base agent using LangGraph's prebuilt create_react_agent.
Replaces custom ReAct implementation with ~150 lines of clean, maintainable code.
"""

import threading
from typing import List, Optional, AsyncGenerator, Dict, Any
from abc import abstractmethod
import asyncio

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
)
from langgraph.prebuilt import create_react_agent

from app.agents.domain.entities import (
    AgentConfiguration,
    AgentRequest,
    AgentResponse,
    AgentExecutionState,
    ExecutionStatus,
    ToolExecution,
)
from app.agents.domain.interfaces import BaseAgent
from app.agents.domain.services import StandardResultProcessor
from app.agents.settings import get_llm
from app.agents.tools.registry import ToolRegistry
from app.agents.schemas import StreamChunk
from app.agents.logging import append_chatbot_log


class TimeoutException(Exception):
    """Raised when execution exceeds timeout."""

    pass


def run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """Run a function with a timeout using threading."""
    result_container = {"result": None, "exception": None, "completed": False}

    def target():
        try:
            result_container["result"] = func(*args, **kwargs)
            result_container["completed"] = True
        except Exception as e:
            result_container["exception"] = e
            result_container["completed"] = True

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if not result_container["completed"]:
        raise TimeoutException(f"Execution exceeded {timeout_seconds}s timeout")

    if result_container["exception"]:
        raise result_container["exception"]

    return result_container["result"]


class BaseLangGraphAgent(BaseAgent):
    """
    Base agent using LangGraph's prebuilt create_react_agent.

    Provides:
    - Automatic ReAct loop (no custom graph needed)
    - Native function calling (no JSON mode)
    - Streaming support with tool-aware status updates
    - Clean message building with extension points
    - Timeout handling
    - Result processing

    Subclasses must implement:
    - _get_system_prompt() -> str
    - _create_tool_registry() -> ToolRegistry
    - _get_target_category() -> str
    """

    def __init__(
        self,
        configuration: AgentConfiguration,
        mode: str = "stream",
        ctx_user=None,
        ctx_db=None,
    ):
        super().__init__(configuration)

        self.mode = mode

        self.llm = get_llm(
            tier=configuration.llm_tier,
            json_mode=(mode == "json"),
            temperature=0.0,
            streaming=(mode == "stream"),
            timeout=configuration.timeout_seconds,
        )

        self.tool_registry = self._create_tool_registry(ctx_user, ctx_db)
        self.tools = self.tool_registry.get_tools()

        if self.configuration.allowed_tools:
            self.tools = [t for t in self.tools if t.name in self.configuration.allowed_tools]

        self.result_processor = StandardResultProcessor()

        # ReAct graph only needed in stream mode
        if mode == "stream":
            llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm
            self.graph = create_react_agent(llm_with_tools, self.tools)
        else:
            self.graph = None

        append_chatbot_log(
            f"Initialized {self.__class__.__name__} with {len(self.tools)} tools (mode={mode})"
        )

    # ============================================================================
    # MESSAGE BUILDING
    # ============================================================================

    def _build_messages(self, request: AgentRequest, **context) -> List[BaseMessage]:
        """
        Build message list for agent execution.

        Standard pattern:
        1. System prompt (always first)
        2. Context messages (extension point for strategy, etc.)
        3. Conversation history (last 3 turns)
        4. Current query (always last)

        Args:
           request: Agent request with query and history
           **context: Additional context passed to _add_context_messages()

        Returns:
           List of messages for the agent
        """
        messages = []

        # 1. System prompt
        messages.append(SystemMessage(content=self._get_system_prompt()))

        # 2. Agent-specific context (e.g., strategy for RetrievalAgent)
        context_msgs = self._add_context_messages(**context)
        messages.extend(context_msgs)

        # 3. Conversation history
        messages.extend(self._convert_history(request.conversation_history))

        # 4. Current query
        messages.append(HumanMessage(content=request.user_text))

        return messages

    def _add_context_messages(self, **context) -> List[BaseMessage]:
        """
        Extension point for adding agent-specific context messages.

        Override to inject additional context between system prompt and history.
        Examples: strategy from planner, current execution state, etc.

        Args:
                        **context: Arbitrary context passed from execute method

        Returns:
           List of context messages (default: empty)
        """
        return []

    def _convert_history(self, history: List[dict]) -> List[BaseMessage]:
        """
        Convert conversation history to message format.

        Args:
           history: List of turns with 'u' (user) and 'a' (assistant) keys

        Returns:
           List of messages (last 3 turns)
        """
        messages = []
        for turn in history[-3:]:
            if "u" in turn:
                messages.append(HumanMessage(content=turn["u"]))
            if "a" in turn:
                messages.append(AIMessage(content=turn["a"]))
        return messages

    # ============================================================================
    # SYNCHRONOUS EXECUTION
    # ============================================================================

    async def execute_json(self, request: AgentRequest, response_model: type, **context) -> Any:
        """
        Execute a single structured LLM call without a ReAct loop.

        Only available in json mode. Builds messages via the standard
        _build_messages() pipeline so subclass context injection works
        identically to stream mode.

        Args:
                        request: Agent request with query and history
                        response_model: Pydantic model to validate and parse the response into
                        **context: Additional context passed to _add_context_messages()

        Returns:
                        Validated instance of response_model

        Raises:
                        RuntimeError: If called on a stream-mode agent
                        ValidationError: If response cannot be parsed into response_model
        """
        if self.mode != "json":
            raise RuntimeError(
                f"{self.__class__.__name__}.execute_json() called but agent is in '{self.mode}' mode"
            )

        messages = self._build_messages(request, **context)

        try:
            response = await self.llm.ainvoke(messages)
            return response_model.model_validate_json(response.content)

        except Exception as e:
            append_chatbot_log(f"[JSON EXECUTE ERROR] {type(e).__name__}: {e}")
            raise

    # ============================================================================
    # STREAMING EXECUTION
    # ============================================================================

    async def execute_stream(
        self, request: AgentRequest, **context
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Execute agent with true token-by-token streaming and tool tracking.

        Uses stream_mode=["messages", "values"] to get both:
          - "messages": (AIMessageChunk, metadata) tuples for real-time token streaming
          - "values":	full state snapshots after each node, used to extract the
        complete tool call history for verification/evals (same data as the old stream_mode="values" loop produced)

        Chunk type contract:
          - status:   progress updates (start, tool begin, tool complete)
          - token:	  individual LLM output delta — only AIMessageChunk with prose content
          - complete: final metadata including tool_calls reconstructed from full state

        Args:
            request: Agent request with query and history
            **context: Additional context passed to message builder

        Yields:
            StreamChunk objects with status updates, tokens, and completion
        """
        if self.mode != "stream":
            raise RuntimeError(
                f"{self.__class__.__name__}.execute_stream() called but agent is in '{self.mode}' mode"
            )

        append_chatbot_log(
            f"\n{'=' * 60}\n{self.__class__.__name__.upper()} STREAMING\n"
            f"Query: {request.user_text[:50]}...\n{'=' * 60}"
        )

        messages = self._build_messages(request, **context)

        state = AgentExecutionState(
            input_text=request.user_text,
            conversation_history=request.conversation_history,
            status=ExecutionStatus.RUNNING,
        )

        try:
            yield StreamChunk(type="status", content=self._get_start_status())

            accumulated_text = []
            tool_in_progress = False
            # Receives the full state snapshot after every node; the last one is
            # post-completion and is equivalent to what stream_mode="values" gave us.
            final_state_snapshot = {}

            async for mode, data in self.graph.astream(
                {"messages": messages},
                stream_mode=["messages", "values"],
            ):
                if mode == "values":
                    # Track latest full state — last one wins after the loop.
                    # This preserves full tool call / ToolMessage history for
                    # _extract_tool_calls_from_messages, identical to the old values loop.
                    final_state_snapshot = data

                elif mode == "messages":
                    msg, metadata = data

                    # --- AIMessageChunk: token-level deltas from the LLM ---
                    # Must use isinstance, not duck-typing: AIMessage (full, non-chunk)
                    # is also emitted in messages mode once the node finishes and would
                    # duplicate the entire response if not explicitly excluded here.
                    if isinstance(msg, AIMessageChunk):
                        has_content = bool(msg.content)
                        has_tool_chunks = bool(msg.tool_call_chunks)

                        if has_tool_chunks and not has_content:
                            # LLM is streaming its tool-call decision.
                            # The tool name is only present on the very first chunk;
                            # subsequent chunks carry argument JSON fragments with name=None.
                            first_chunk = msg.tool_call_chunks[0]
                            tool_name = first_chunk.get("name") or ""
                            if tool_name and not tool_in_progress:
                                # args are not available yet — they stream in subsequent
                                # chunks — so _get_tool_start_status receives {} here.
                                status = self._get_tool_start_status(tool_name, {})
                                yield StreamChunk(type="status", content=status)
                                tool_in_progress = True

                        elif has_content and not has_tool_chunks:
                            # Prose delta — this is the content users read.
                            token = str(msg.content)
                            accumulated_text.append(token)
                            yield StreamChunk(type="token", content=token)

                    # --- ToolMessage: tool execution has completed ---
                    elif hasattr(msg, "tool_call_id"):
                        if tool_in_progress:
                            yield StreamChunk(
                                type="status", content=self._get_tool_complete_status()
                            )
                            tool_in_progress = False

            # ------------------------------------------------------------------
            # Post-loop: build completion chunk from final state snapshot.
            # final_state_snapshot["messages"] is the identical full message list
            # that stream_mode="values" used to populate all_messages — no changes
            # needed to _extract_tool_calls_from_messages or any caller.
            # ------------------------------------------------------------------
            all_messages = final_state_snapshot.get("messages", [])
            tool_calls = self._extract_tool_calls_from_messages(all_messages)

            state.mark_completed()

            if not accumulated_text:
                fallback = "I couldn't generate a response."
                accumulated_text = [fallback]
                yield StreamChunk(type="token", content=fallback)

            yield StreamChunk(
                type="complete",
                data={
                    "target": self._get_target_category(),
                    "success": True,
                    "book_ids": [],
                    "tool_calls": tool_calls,
                    "citations": [],
                    "policy_version": self.configuration.policy_name,
                    "elapsed_ms": state.execution_time_ms,
                },
            )

        except Exception as e:
            append_chatbot_log(f"[STREAMING ERROR] {type(e).__name__}: {e}")
            state.mark_failed(str(e))

            yield StreamChunk(
                type="complete",
                data={
                    "target": self._get_target_category(),
                    "success": False,
                    "text": "I encountered an error processing your request.",
                    "error": str(e),
                    "tool_calls": [],
                    "policy_version": self.configuration.policy_name,
                    "elapsed_ms": state.execution_time_ms,
                },
            )

    def _extract_tool_calls_from_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Extract tool calls with results from message history.

        Args:
            messages: List of messages from graph execution

        Returns:
            List of dicts with tool_name, arguments, result
        """
        # Build mapping of tool_call_id -> result
        tool_results = {}
        for msg in messages:
            if hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
                tool_results[msg.tool_call_id] = msg.content

        # Extract tool calls and match with results
        tool_calls = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_call_id = tool_call.get("id", "")
                    tool_result = tool_results.get(tool_call_id, "")

                    tool_calls.append(
                        {
                            "tool_name": tool_call.get("name", ""),
                            "arguments": tool_call.get("args", {}),
                            "result": tool_result,
                        }
                    )

        return tool_calls

    def _tool_execution_to_dict(self, te: ToolExecution) -> Dict:
        """Convert ToolExecution to dict for serialization."""
        return {
            "tool_name": te.tool_name,
            "arguments": te.arguments,
            "result": te.result,
        }

    # ============================================================================
    # STREAMING CUSTOMIZATION HOOKS
    # ============================================================================

    def _get_start_status(self) -> str:
        """
        Get initial status message when execution starts.
        Override to customize initial status per agent.
        """
        return "Processing request..."

    def _get_tool_start_status(self, tool_name: str, args: dict) -> str:
        """
        Get status message when tool is about to execute.

        Automatically uses tool.metadata["status_message"] if available,
        with support for {arg_name} placeholders.

        Override for additional customization beyond tool metadata.

        Args:
            tool_name: Name of tool being called
            args: Arguments passed to tool

        Returns:
            Status message to display
        """
        # Find tool object by name
        tool = next((t for t in self.tools if t.name == tool_name), None)

        # Use custom status_message from metadata if available
        if tool and hasattr(tool, "metadata") and isinstance(tool.metadata, dict):
            status_msg = tool.metadata.get("status_message")
            if status_msg:
                try:
                    # Format with args (e.g., "Reading {doc_name}..." → "Reading intro.md...")
                    return status_msg.format(**args)
                except (KeyError, ValueError):
                    # If formatting fails, use message as-is
                    return status_msg

        # Default fallback
        return f"Using {tool_name}..."

    def _get_tool_complete_status(self) -> str:
        """
        Get status message after tool completes.
        Override to customize per agent.
        """
        return "Processing results..."

    # ============================================================================
    # ABSTRACT METHODS (subclasses must implement)
    # ============================================================================

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Get system prompt for this agent.

        Returns:
            System prompt as string
        """
        pass

    @abstractmethod
    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        """
        Create tool registry for this agent.

        Args:
            ctx_user: User context (optional)
            ctx_db: Database session (optional)

        Returns:
            Configured ToolRegistry instance
        """
        pass

    @abstractmethod
    def _get_target_category(self) -> str:
        """
        Get target category for this agent (e.g., "docs", "recsys", "respond").

        Returns:
            Target category string
        """
        pass

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _extract_tool_call(self, event: dict) -> Optional[dict]:
        """
        Extract tool call information from agent event.

        Args:
            event: Event dict from graph.astream()

        Returns:
            Tool call dict with 'name' and 'args', or None
        """
        messages = event.get("agent", {}).get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return last_msg.tool_calls[0]

        return None

    def _extract_final_text(self, event: dict) -> Optional[str]:
        """
        Extract final response text from agent event.

        Args:
            event: Event dict from graph.astream()

        Returns:
            Response text, or None
        """
        messages = event.get("agent", {}).get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if hasattr(last_msg, "content") and last_msg.content:
            content = str(last_msg.content).strip()
            if content and len(content) > 10:
                return content

        return None

    def _tokenize_for_streaming(self, text: str) -> List[str]:
        """
        Tokenize text for word-by-word streaming.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (words with spaces)
        """
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            # Add space after all words except the last
            token = word if i == len(words) - 1 else word + " "
            tokens.append(token)
        return tokens
