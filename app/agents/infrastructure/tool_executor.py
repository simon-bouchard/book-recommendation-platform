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


class ToolExecutor:
    """
    Executes native tools and wraps results in domain entities.
    
    Key improvements over legacy executor:
    - Works with typed function signatures
    - No string/JSON parsing gymnastics
    - Clear error messages
    - Automatic parameter validation
    """
    
    def __init__(self, registry: ModernToolRegistry):
        """
        Initialize executor with a tool registry.
        
        Args:
            registry: Tool registry providing available tools
        """
        self.registry = registry
        self._tools: Dict[str, ToolDefinition] = {
            t.name: t for t in registry.get_tools()
        }
    
    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ToolExecution:
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
                execution_time_ms=0
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
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return ToolExecution(
                tool_name=tool_name,
                arguments=arguments,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=execution_time_ms
            )
    
    def _prepare_arguments(
        self,
        tool: ToolDefinition,
        raw_arguments: Dict[str, Any]
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
            if param_name in ('self', 'cls'):
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
                            f"Cannot convert {param_name}={value!r} to "
                            f"{param.annotation}: {e}"
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
        
        Handles common conversions:
        - str to int/float/bool
        - JSON strings to list/dict
        - etc.
        
        Args:
            value: Value to convert
            target_type: Target type
            
        Returns:
            Converted value
            
        Raises:
            ValueError: If conversion fails
        """
        # Already correct type
        if isinstance(value, target_type):
            return value
        
        # String to numeric
        if target_type in (int, float) and isinstance(value, str):
            return target_type(value)
        
        # String to bool
        if target_type == bool and isinstance(value, str):
            lower = value.lower()
            if lower in ('true', '1', 'yes'):
                return True
            elif lower in ('false', '0', 'no'):
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
                f"to {target_type.__name__}"
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
                if 'book_ids' in result:
                    book_ids = [int(x) for x in result['book_ids']]
                elif 'item_idx' in result:
                    book_ids = [int(result['item_idx'])]
            
            # List of book objects or IDs
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # Try item_idx (standard) or book_id (fallback)
                        id_val = item.get('item_idx') or item.get('book_id')
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
            'book_semantic_search',
            'als_recs',
            'subject_hybrid_pool',
            'subject_id_search',
            'return_book_ids',
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
            if name in ('self', 'cls'):
                continue
            
            param_info = {'required': param.default == inspect.Parameter.empty}
            
            if param.annotation != inspect.Parameter.empty:
                param_info['type'] = getattr(
                    param.annotation, '__name__',
                    str(param.annotation)
                )
            
            if param.default != inspect.Parameter.empty:
                param_info['default'] = param.default
            
            params[name] = param_info
        
        return {
            'name': tool.name,
            'description': tool.description,
            'category': tool.category.value,
            'requires_auth': tool.metadata.requires_auth,
            'requires_db': tool.metadata.requires_db,
            'parameters': params,
        }