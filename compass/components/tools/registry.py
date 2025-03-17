"""Tool registry for managing collections of tools."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable

from compass.components.tools.tool import Tool
from compass.components.tools.function_tool import FunctionTool


@dataclass
class ToolRegistry:
    """
    A registry for managing a collection of tools.
    
    The registry maintains a set of tools that can be used by agents or models,
    and provides methods for registering, retrieving, and formatting tools.
    """
    
    tools: Dict[str, Tool] = field(default_factory=dict)
    """Dictionary of registered tools, keyed by tool name."""
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: The tool to register.
            
        Raises:
            ValueError: If a tool with the same name already exists.
        """
        if tool.name in self.tools:
            raise ValueError(f"A tool with name '{tool.name}' is already registered")
        
        self.tools[tool.name] = tool
    
    def register_function(self, func: Optional[Callable] = None, *, 
                         name: Optional[str] = None, 
                         description: Optional[str] = None) -> Union[FunctionTool, Callable]:
        """
        Register a function as a tool.
        
        Can be used as a decorator or as a regular function.
        
        Args:
            func: The function to register.
            name: Optional name for the tool. If not provided, uses the function name.
            description: Optional description for the tool. If not provided, uses the function docstring.
            
        Returns:
            Union[FunctionTool, Callable]: The FunctionTool if called directly, or a decorator if used as a decorator.
        """
        def _register(f: Callable) -> FunctionTool:
            tool = FunctionTool(
                func=f,
                name=name or f.__name__,
                description=description or f.__doc__ or f"Function {f.__name__}"
            )
            self.register_tool(tool)
            return tool
        
        if func is None:
            # Used as a decorator with arguments
            return _register
        else:
            # Used as a decorator without arguments or called directly
            return _register(func)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: The name of the tool to retrieve.
            
        Returns:
            Optional[Tool]: The tool if found, None otherwise.
        """
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            List[Tool]: List of all registered tools.
        """
        return list(self.tools.values())
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all registered tools to OpenAI's tool format.
        
        Returns:
            List[Dict[str, Any]]: List of tools in OpenAI's format.
        """
        return [tool.to_openai_tool() for tool in self.tools.values()]
    
    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all registered tools to Anthropic's tool format.
        
        Returns:
            List[Dict[str, Any]]: List of tools in Anthropic's format.
        """
        return [tool.to_anthropic_tool() for tool in self.tools.values()]