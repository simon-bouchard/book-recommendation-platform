# app/agents/utils/retrieval_logging_callback.py
"""
Callback handler for logging retrieval tool executions in real-time.
"""

import json
import time
from typing import Any, Dict, List, Optional
from langchain.callbacks.base import BaseCallbackHandler

from app.agents.logging import append_chatbot_log, is_debug_mode


class RetrievalLoggingCallback(BaseCallbackHandler):
    """
    Real-time logging callback for retrieval agent tool executions.

    Logs:
    - Tool start with arguments
    - Tool end with duration, result count, and cumulative count
    - Sample results in debug mode
    - Tool errors
    """

    def __init__(self):
        """Initialize callback with tracking state."""
        super().__init__()
        self.tool_count = 0
        self.accumulated_books = 0
        self.current_tool_name: Optional[str] = None
        self.tool_start_time: Optional[float] = None

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """
        Called when tool starts executing.

        Args:
                                        serialized: Tool metadata including name
                                        input_str: Tool input (JSON string of arguments)
                                        **kwargs: Additional metadata
        """
        self.tool_count += 1
        self.current_tool_name = serialized.get("name", "unknown")
        self.tool_start_time = time.time()

        # Parse and format arguments
        args_str = self._format_tool_input(input_str)

        append_chatbot_log(f"[TOOL {self.tool_count}] {self.current_tool_name}({args_str})")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # Change type hint from str to Any
        """Called when tool completes successfully."""

        # Extract content from ToolMessage if needed
        if hasattr(output, "content"):
            output_str = output.content
        else:
            output_str = str(output)

        # Calculate duration
        duration_ms = 0
        if self.tool_start_time:
            duration_ms = int((time.time() - self.tool_start_time) * 1000)

        # Count books in output
        book_count = self._count_books_in_output(output_str)  # Use output_str
        self.accumulated_books += book_count

        # Log result summary
        append_chatbot_log(f"		  → {book_count} books returned ({duration_ms}ms)")
        append_chatbot_log(f"		  → Cumulative: {self.accumulated_books} candidates")

        # Debug mode: show sample results
        if is_debug_mode():
            sample = self._extract_sample_books(output_str, limit=5)  # Use output_str
            if sample:
                sample_str = ", ".join(sample)
                append_chatbot_log(f"		  Sample: {sample_str}")

        # Reset tool tracking
        self.current_tool_name = None
        self.tool_start_time = None

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """
        Called when tool execution fails.

        Args:
                                        error: Exception that occurred
                                        **kwargs: Additional metadata
        """
        error_msg = str(error)
        if len(error_msg) > 100:
            error_msg = error_msg[:97] + "..."

        append_chatbot_log(f"		  ✗ Error: {error_msg}")

        # Reset tool tracking
        self.current_tool_name = None
        self.tool_start_time = None

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _format_tool_input(self, input_str: str) -> str:
        """
        Format tool input arguments for logging.

        Args:
                                        input_str: JSON string of tool arguments

        Returns:
                                        Formatted argument string (e.g., "query='fantasy', limit=100")
        """
        try:
            args = json.loads(input_str)

            if isinstance(args, dict):
                # Format as key=value pairs
                formatted_args = []
                for key, value in args.items():
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    elif isinstance(value, list) and len(value) > 5:
                        value = f"[{len(value)} items]"

                    formatted_args.append(f"{key}={repr(value)}")

                return ", ".join(formatted_args)
            else:
                # Non-dict input, just show truncated
                return str(args)[:100]

        except json.JSONDecodeError:
            # Not JSON, show truncated string
            return input_str[:100] if len(input_str) > 100 else input_str

    def _count_books_in_output(self, output: str) -> int:
        """Count books in tool output."""
        if not output or not isinstance(output, str):
            return 0

        try:
            # Parse JSON (output.content is a JSON string)
            data = json.loads(output)

            # Handle list of books directly
            if isinstance(data, list):
                # Filter out error dicts
                books = [item for item in data if isinstance(item, dict) and "error" not in item]
                return len(books)

            # Handle dict wrapper
            if isinstance(data, dict):
                for key in ["books", "results", "items", "candidates"]:
                    if key in data and isinstance(data[key], list):
                        return len(data[key])

            return 0

        except json.JSONDecodeError:
            return 0
        except Exception:
            return 0

    def _extract_sample_books(self, output: str, limit: int = 5) -> List[str]:
        """Extract sample book titles from tool output."""
        if not output or not isinstance(output, str):
            return []

        try:
            # Parse JSON
            data = json.loads(output)

            # Get books list
            books = []
            if isinstance(data, list):
                books = [item for item in data if isinstance(item, dict) and "error" not in item]
            elif isinstance(data, dict):
                for key in ["books", "results", "items", "candidates"]:
                    if key in data and isinstance(data[key], list):
                        books = data[key]
                        break

            if not books:
                return []

            # Extract compact representations - NO truncation in debug mode
            sample = []
            for book in books[:limit]:
                if isinstance(book, dict):
                    item_idx = book.get("item_idx", "?")
                    title = book.get("title", "Unknown")

                    # Show full title in debug mode
                    sample.append(f"{item_idx}: {title}")

            return sample

        except json.JSONDecodeError:
            return []
        except Exception:
            return []
