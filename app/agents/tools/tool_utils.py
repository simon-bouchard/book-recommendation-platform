# app/agents/tools/tool_utils.py
"""
Utility decorator for adding streaming status messages to LangChain tools.
"""

from typing import Callable, Optional

from langchain_core.tools import tool as langchain_tool


def tool_with_status(status_message: Optional[str] = None):
    """
    Decorator that creates a LangChain tool with optional streaming status message.

    The status_message can include {arg_name} placeholders that will be formatted
    with the tool's arguments during execution.

    Usage:
        @tool_with_status("Reading {doc_name}...")
        def help_read(doc_name: str) -> str:
            '''Read a help document.'''
            return read_document(doc_name)

    Args:
        status_message: Optional status message with {arg} placeholders

    Returns:
        Decorated function as a LangChain tool with status_message attribute
    """

    def decorator(func: Callable) -> Callable:
        # Create LangChain tool
        tool_func = langchain_tool(func)

        # Add status_message attribute if provided
        if status_message:
            tool_func.status_message = status_message

        return tool_func

    return decorator


def add_status_message(tool_func: Callable, status_message: str) -> Callable:
    """
    Add status_message attribute to an existing LangChain tool.

    Useful when you can't modify the tool creation code but want to add
    streaming status messages.

    Usage:
        @tool
        def my_tool(arg: str) -> str:
            return result

        my_tool = add_status_message(my_tool, "Processing {arg}...")

    Args:
        tool_func: Existing LangChain tool
        status_message: Status message with optional {arg} placeholders

    Returns:
        Tool with status_message attribute added
    """
    tool_func.status_message = status_message
    return tool_func
