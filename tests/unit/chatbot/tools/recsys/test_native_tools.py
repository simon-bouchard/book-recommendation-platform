# tests/unit/chatbot/tools/test_native_tool.py
"""
Tests for the native tool infrastructure (decorators, metadata, definitions).
Validates that the tool system works correctly without external dependencies.
"""

import pytest
import inspect
from app.agents.tools.native_tool import (
    tool,
    ToolCategory,
    ToolMetadata,
    ToolDefinition,
)


class TestToolDecorator:
    """Test the @tool decorator creates ToolDefinition correctly."""

    def test_creates_tool_definition(self):
        """Decorator should wrap function in ToolDefinition."""

        @tool(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.INTERNAL,
        )
        def my_tool(arg: str) -> str:
            return f"result: {arg}"

        assert isinstance(my_tool, ToolDefinition)
        assert my_tool.name == "test_tool"
        assert my_tool.description == "A test tool"
        assert my_tool.category == ToolCategory.INTERNAL

    def test_preserves_function_signature(self):
        """Decorator should preserve original function signature."""

        @tool(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.INTERNAL,
        )
        def my_tool(arg1: str, arg2: int = 10) -> str:
            return f"{arg1}: {arg2}"

        sig = my_tool.get_signature()
        params = list(sig.parameters.keys())

        assert "arg1" in params
        assert "arg2" in params
        assert sig.parameters["arg2"].default == 10

    def test_function_is_callable(self):
        """Decorated tool should be executable."""

        @tool(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.INTERNAL,
        )
        def my_tool(arg: str) -> str:
            return f"result: {arg}"

        result = my_tool.execute(arg="test")
        assert result == "result: test"

    def test_sets_metadata_flags(self):
        """Decorator should set all metadata flags correctly."""

        @tool(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.WEB,
            requires_auth=True,
            requires_db=True,
            is_deterministic=False,
        )
        def my_tool():
            pass

        assert my_tool.metadata.requires_auth is True
        assert my_tool.metadata.requires_db is True
        assert my_tool.metadata.is_deterministic is False

    def test_default_metadata_flags(self):
        """Decorator should use sensible defaults for optional flags."""

        @tool(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.INTERNAL,
        )
        def my_tool():
            pass

        assert my_tool.metadata.requires_auth is False
        assert my_tool.metadata.requires_db is False
        assert my_tool.metadata.is_deterministic is True


class TestToolMetadata:
    """Test ToolMetadata dataclass behavior."""

    def test_to_llm_description(self):
        """Metadata should format correctly for LLM consumption."""
        metadata = ToolMetadata(
            name="book_search",
            description="Search for books by title",
            category=ToolCategory.INTERNAL,
        )

        result = metadata.to_llm_description()
        assert "book_search" in result
        assert "Search for books by title" in result


class TestToolDefinition:
    """Test ToolDefinition functionality."""

    def test_executes_with_kwargs(self):
        """ToolDefinition should execute function with keyword arguments."""

        def sample_func(a: int, b: int) -> int:
            return a + b

        metadata = ToolMetadata(
            name="add",
            description="Add two numbers",
            category=ToolCategory.INTERNAL,
        )

        tool_def = ToolDefinition(func=sample_func, metadata=metadata)
        result = tool_def.execute(a=5, b=3)

        assert result == 8

    def test_get_signature_returns_correct_signature(self):
        """get_signature should return function's signature."""

        def sample_func(a: int, b: str = "default") -> str:
            return f"{a}: {b}"

        metadata = ToolMetadata(
            name="test",
            description="Test function",
            category=ToolCategory.INTERNAL,
        )

        tool_def = ToolDefinition(func=sample_func, metadata=metadata)
        sig = tool_def.get_signature()

        assert isinstance(sig, inspect.Signature)
        assert "a" in sig.parameters
        assert "b" in sig.parameters
        assert sig.parameters["b"].default == "default"

    def test_to_llm_description_includes_signature(self):
        """to_llm_description should include parameter types."""

        def sample_func(query: str, limit: int) -> list:
            return []

        metadata = ToolMetadata(
            name="search",
            description="Search function",
            category=ToolCategory.INTERNAL,
        )

        tool_def = ToolDefinition(func=sample_func, metadata=metadata)
        result = tool_def.to_llm_description()

        assert "search" in result
        assert "query" in result
        assert "limit" in result
        assert "Search function" in result

    def test_properties_access_metadata(self):
        """Properties should provide convenient access to metadata fields."""
        metadata = ToolMetadata(
            name="test_tool",
            description="A test",
            category=ToolCategory.WEB,
        )

        tool_def = ToolDefinition(func=lambda: None, metadata=metadata)

        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test"
        assert tool_def.category == ToolCategory.WEB


class TestToolCategory:
    """Test ToolCategory enum."""

    def test_categories_exist(self):
        """All expected categories should be defined."""
        assert ToolCategory.WEB.value == "web"
        assert ToolCategory.DOCS.value == "docs"
        assert ToolCategory.INTERNAL.value == "internal"

    def test_categories_are_unique(self):
        """Category values should be distinct."""
        values = [c.value for c in ToolCategory]
        assert len(values) == len(set(values))
