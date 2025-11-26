# app/agents/infrastructure/tool_executor.py
"""
Modernized tool executor working with native tools.
Clean, type-safe execution without string parsing gymnastics.
"""

import time
import json
import inspect
from typing import Dict, Any, Optional

from app.agents.domain.entities import ToolExecution
from app.agents.tools.registry import ToolRegistry
from app.agents.tools.native_tool import ToolDefinition
from app.agents.logging import append_chatbot_log


class ToolExecutor:
    """
    Executes native tools and wraps results in domain entities.

    Key improvements over legacy executor:
    - Works with typed function signatures
    - No string/JSON parsing gymnastics
    - Clear error messages
    - Automatic parameter validation
    """

    def __init__(self, registry: ToolRegistry):
        """
        Initialize executor with a tool registry.

        Args:
            registry: Tool registry providing available tools
        """
        self.registry = registry
        self._tools: Dict[str, ToolDefinition] = {t.name: t for t in registry.get_tools()}

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecution:
        """
        Execute a tool with given arguments.

        Args:
            tool_name: Name of tool to execute
            arguments: Dictionary of arguments to pass

        Returns:
            ToolExecution domain entity with result or error
        """
        start_time = time.time()

        # Check if tool exists
        if tool_name not in self._tools:
            return ToolExecution(
                tool_name=tool_name,
                arguments=arguments,
                error=f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}",
                execution_time_ms=0,
            )

        tool = self._tools[tool_name]
        try:
            # Validate and prepare arguments
            prepared_args = self._prepare_arguments(tool, arguments)

            # Execute the tool
            result = tool.execute(**prepared_args)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return ToolExecution(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"{type(e).__name__}: {str(e)}"
            append_chatbot_log(f"Tool error: {error_msg}")  # ADD THIS
            return ToolExecution(
                tool_name=tool_name,
                arguments=arguments,
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )

    def _prepare_arguments(
        self, tool: ToolDefinition, raw_arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare and validate arguments for tool execution.

        Handles:
        - Type conversion where safe
        - Default values
        - Missing required parameters
        - Extra parameters

        Args:
            tool: Tool definition with function signature
            raw_arguments: Raw arguments from caller

        Returns:
            Validated and prepared arguments

        Raises:
            ValueError: If required parameters missing or types invalid
        """
        sig = tool.get_signature()
        prepared = {}

        for param_name, param in sig.parameters.items():
            # Skip self/cls parameters
            if param_name in ("self", "cls"):
                continue

            # Check if argument provided
            if param_name in raw_arguments:
                value = raw_arguments[param_name]

                # Attempt type coercion if annotation available
                if param.annotation != inspect.Parameter.empty:
                    try:
                        value = self._coerce_type(value, param.annotation)
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Cannot convert {param_name}={value!r} to {param.annotation}: {e}"
                        )

                prepared[param_name] = value

            elif param.default != inspect.Parameter.empty:
                # Use default value
                prepared[param_name] = param.default

            else:
                # Required parameter missing
                raise ValueError(
                    f"Missing required parameter '{param_name}' for tool '{tool.name}'"
                )

        return prepared

    def _coerce_type(self, value: Any, target_type: type) -> Any:
        """
        Attempt to coerce value to target type.

        Handles parameterized generics like list[int], dict[str, Any], Optional[X], etc.
        """
        import typing

        # Handle Optional[X] (which is Union[X, None])
        origin = typing.get_origin(target_type)

        if origin is typing.Union:
            # Get the non-None type from Union
            args = typing.get_args(target_type)
            non_none_types = [arg for arg in args if arg is not type(None)]

            # If value is None and None is allowed, that's fine
            if value is None and type(None) in args:
                return None

            # Try to coerce to the first non-None type
            if non_none_types:
                return self._coerce_type(value, non_none_types[0])

            raise ValueError(f"Cannot coerce {value!r} to {target_type}")

        # Handle other parameterized generics (list[int], dict[str, Any], etc.)
        if origin is not None:
            # For list[int], dict[str, Any], etc. - check the origin type only
            if isinstance(value, origin):
                return value

            # Try to convert if needed
            if origin in (list, dict) and isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, origin):
                        return parsed
                except json.JSONDecodeError:
                    pass

            raise ValueError(
                f"Cannot convert {value!r} (type {type(value).__name__}) to {origin.__name__}"
            )

        # Non-generic types - safe to use isinstance
        if isinstance(value, target_type):
            return value

        # String to numeric
        if target_type in (int, float) and isinstance(value, str):
            return target_type(value)

        # String to bool
        if target_type == bool and isinstance(value, str):
            lower = value.lower()
            if lower in ("true", "1", "yes"):
                return True
            elif lower in ("false", "0", "no"):
                return False
            raise ValueError(f"Cannot convert '{value}' to bool")

        # JSON string to list/dict
        if target_type in (list, dict) and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, target_type):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Generic conversion attempt
        try:
            return target_type(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Cannot convert {value!r} (type {type(value).__name__}) "
                f"to {getattr(target_type, '__name__', str(target_type))}"
            )

    def extract_book_ids_from_result(self, result: Any) -> list[int]:
        """
        Extract book IDs from a tool result.

        Handles various result formats:
        - dict with 'book_ids' key
        - list of dicts with 'item_idx' keys
        - list of integers

        Args:
            result: Tool execution result

        Returns:
            List of book IDs (item_idx values)
        """
        book_ids = []

        try:
            # Dictionary with book_ids key
            if isinstance(result, dict):
                if "book_ids" in result:
                    book_ids = [int(x) for x in result["book_ids"]]
                elif "item_idx" in result:
                    book_ids = [int(result["item_idx"])]

            # List of book objects or IDs
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # Try item_idx (standard) or book_id (fallback)
                        id_val = item.get("item_idx") or item.get("book_id")
                        if id_val is not None:
                            book_ids.append(int(id_val))
                    elif isinstance(item, int):
                        book_ids.append(item)

        except (ValueError, TypeError, KeyError):
            pass

        return book_ids

    def is_book_recommendation_tool(self, tool_name: str) -> bool:
        """
        Check if a tool returns book recommendations.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool returns book recommendations
        """
        recommendation_tools = {
            "book_semantic_search",
            "als_recs",
            "subject_hybrid_pool",
            "subject_id_search",
            "return_book_ids",
            "popular_books",
        }
        return tool_name.lower() in recommendation_tools

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool.

        Args:
            tool_name: Name of tool

        Returns:
            Dictionary with tool metadata, or None if not found
        """
        if tool_name not in self._tools:
            return None

        tool = self._tools[tool_name]
        sig = tool.get_signature()

        params = {}
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            param_info = {"required": param.default == inspect.Parameter.empty}

            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = getattr(param.annotation, "__name__", str(param.annotation))

            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            params[name] = param_info

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "requires_auth": tool.metadata.requires_auth,
            "requires_db": tool.metadata.requires_db,
            "parameters": params,
        }

    def extract_books_from_result(self, result: Any) -> list[dict]:
        """
        Extract full book objects (not just IDs) from tool result.

        Returns list of dicts with fields like:
        - item_idx (required)
        - title, author, year (optional)
        - subjects, tones, genre (optional)
        - description (optional)
        - score (optional)

        Args:
            result: Tool execution result

        Returns:
            List of book dictionaries with available metadata
        """
        books = []

        try:
            # Handle dict with 'books' or 'results' key
            if isinstance(result, dict):
                book_list = result.get("books") or result.get("results") or []
                if isinstance(book_list, list):
                    books.extend(book_list)
                elif "item_idx" in result:
                    # Single book dict
                    books.append(result)

            # Handle list of book dicts or IDs
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # Book dict - take as is
                        if "item_idx" in item or "book_id" in item:
                            books.append(item)
                    elif isinstance(item, int):
                        # Just an ID - create minimal dict
                        books.append({"item_idx": item})

            # Normalize field names (book_id -> item_idx)
            for book in books:
                if "book_id" in book and "item_idx" not in book:
                    book["item_idx"] = book.pop("book_id")

        except (ValueError, TypeError, KeyError):
            pass

        return books

