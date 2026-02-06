# app/agents/infrastructure/base_langgraph_agent.py
"""
Base agent using LangGraph's prebuilt create_react_agent.
Replaces custom ReAct implementation with ~150 lines of clean, maintainable code.
"""

import time
import threading
from typing import List, Optional, AsyncGenerator
from abc import abstractmethod

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from app.agents.domain.entities import (
    AgentConfiguration,
    AgentRequest,
    AgentResponse,
    AgentExecutionState,
    ExecutionStatus,
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

    def __init__(self, configuration: AgentConfiguration, ctx_user=None, ctx_db=None):
        """
        Initialize base agent.

        Args:
            configuration: Agent configuration with LLM tier, timeout, etc.
            ctx_user: User context for tools (optional)
            ctx_db: Database session for tools (optional)
        """
        super().__init__(configuration)

        # Initialize LLM (native function calling, no JSON mode)
        self.llm = get_llm(tier=configuration.llm_tier, temperature=0.0)

        # Initialize tool registry and get native tool functions
        self.tool_registry = self._create_tool_registry(ctx_user, ctx_db)
        self.tools = self.tool_registry.get_tools()

        if self.configuration.allowed_tools:
            self.tools = [t for t in self.tools if t.name in self.configuration.allowed_tools]

        # Result processor
        self.result_processor = StandardResultProcessor()

        llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

        # Create ReAct agent using prebuilt
        self.graph = create_react_agent(llm_with_tools, self.tools)

        append_chatbot_log(
            f"Initialized {self.__class__.__name__} with {len(self.tools)} tools "
            f"using prebuilt ReAct agent"
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

    def execute(self, request: AgentRequest, **context) -> AgentResponse:
        """
        Execute agent synchronously.

        Args:
            request: Agent request with query and history
            **context: Additional context passed to message builder

        Returns:
            Agent response with result
        """
        append_chatbot_log(
            f"\n{'=' * 60}\n"
            f"{self.__class__.__name__.upper()} EXECUTION\n"
            f"Query: {request.user_text[:100]}...\n"
            f"{'=' * 60}"
        )

        # Build messages
        messages = self._build_messages(request, **context)

        # Create execution state for tracking
        state = AgentExecutionState(
            input_text=request.user_text,
            conversation_history=request.conversation_history,
            status=ExecutionStatus.RUNNING,
        )

        try:
            # Execute with timeout
            result = run_with_timeout(
                self.graph.invoke, self.configuration.timeout_seconds, {"messages": messages}
            )

            # Extract final message
            final_messages = result.get("messages", [])
            if final_messages:
                final_text = final_messages[-1].content
            else:
                final_text = "I couldn't generate a response."

            state.mark_completed()

            # Build response
            response = AgentResponse(
                text=final_text,
                target_category=self._get_target_category(),
                success=True,
                execution_state=state,
                policy_version=self.configuration.policy_name,
            )

            append_chatbot_log(f"COMPLETE: {state.execution_time_ms}ms\n{'=' * 60}\n")

            return response

        except TimeoutException as e:
            append_chatbot_log(f"[TIMEOUT] {e}")
            state.mark_failed(str(e))

            return AgentResponse(
                text="I'm taking too long to respond. Please try rephrasing your request.",
                target_category=self._get_target_category(),
                success=False,
                execution_state=state,
                policy_version=self.configuration.policy_name,
            )

        except Exception as e:
            append_chatbot_log(f"[ERROR] {type(e).__name__}: {e}")
            state.mark_failed(str(e))

            return AgentResponse(
                text="I encountered an error processing your request.",
                target_category=self._get_target_category(),
                success=False,
                execution_state=state,
                policy_version=self.configuration.policy_name,
            )

    # ============================================================================
    # STREAMING EXECUTION
    # ============================================================================

    async def execute_stream(
        self, request: AgentRequest, **context
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Execute agent with streaming output.

        Yields StreamChunks with types:
        - "status": Progress updates (automatic tool detection)
        - "token": Response text tokens
        - "complete": Final result with metadata

        Args:
            request: Agent request with query and history
            **context: Additional context passed to message builder

        Yields:
            StreamChunk objects representing progress
        """
        append_chatbot_log(
            f"\n{'=' * 60}\n"
            f"{self.__class__.__name__.upper()} STREAMING\n"
            f"Query: {request.user_text[:100]}...\n"
            f"{'=' * 60}"
        )

        # Yield initial status
        yield StreamChunk(type="status", content=self._get_start_status())

        # Build messages
        messages = self._build_messages(request, **context)

        # Create execution state
        state = AgentExecutionState(
            input_text=request.user_text,
            conversation_history=request.conversation_history,
            status=ExecutionStatus.RUNNING,
        )

        accumulated_text = []

        try:
            # Stream from agent
            async for event in self.graph.astream({"messages": messages}):
                # Agent output (reasoning or tool calls)
                if "agent" in event:
                    agent_messages = event["agent"].get("messages", [])

                    if agent_messages:
                        last_msg = agent_messages[-1]

                        # Tool about to be called
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            tool_name = last_msg.tool_calls[0].get("name", "")
                            args = last_msg.tool_calls[0].get("args", {})

                            # Get tool-specific status (uses tool.status_message if available)
                            status = self._get_tool_start_status(tool_name, args)
                            yield StreamChunk(type="status", content=status)

                        # Final response text
                        elif hasattr(last_msg, "content") and last_msg.content:
                            content = str(last_msg.content).strip()

                            # Only stream if substantial content
                            if content and len(content) > 10:
                                yield StreamChunk(type="status", content="Generating response...")

                                # Stream tokens word-by-word
                                words = content.split()
                                for i, word in enumerate(words):
                                    token = word if i == len(words) - 1 else word + " "
                                    yield StreamChunk(type="token", content=token)
                                    accumulated_text.append(token)

                # Tool execution completed
                elif "tools" in event:
                    status = self._get_tool_complete_status()
                    yield StreamChunk(type="status", content=status)

            # Mark success
            state.mark_completed()

            # Fallback if no response
            if not accumulated_text:
                accumulated_text = ["I couldn't generate a response."]
                for token in accumulated_text:
                    yield StreamChunk(type="token", content=token)

            # Yield completion
            yield StreamChunk(
                type="complete",
                data={
                    "target": self._get_target_category(),
                    "success": True,
                    "book_ids": [],  # Subclass can override to extract books
                    "tool_calls": [],
                    "citations": [],
                    "policy_version": self.configuration.policy_name,
                    "elapsed_ms": state.execution_time_ms,
                },
            )

        except Exception as e:
            append_chatbot_log(f"[STREAMING ERROR] {type(e).__name__}: {e}")
            state.mark_failed(str(e))

            # Yield error completion
            yield StreamChunk(
                type="complete",
                data={
                    "target": self._get_target_category(),
                    "success": False,
                    "text": "I encountered an error processing your request.",
                    "error": str(e),
                },
            )

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
