"""Function Tool implementation for Compass."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union, TypedDict, Type, get_type_hints
import inspect
import json

from compass.components.tools.tool import Tool, ToolParameter


@dataclass
class FunctionTool(Tool):
    """
    A tool that wraps a Python function.
    
    This tool allows any Python function to be used as a tool, automatically
    generating the appropriate schema from the function's signature and docstring.
    """
    
    func: Callable
    """The function to wrap as a tool."""
    
    name: str = ""
    """The name of the tool. If not provided, uses the function's name."""
    
    description: str = ""
    """A description of what the tool does. If not provided, uses the function's docstring."""
    
    parameters: List[ToolParameter] = field(default_factory=list)
    """The parameters for the tool. If not provided, generated from the function signature."""
    
    def __post_init__(self) -> None:
        """Initialize the tool from the function."""
        # If name not provided, use the function name
        if not self.name:
            self.name = self.func.__name__
        
        # If description not provided, use the function docstring
        if not self.description:
            self.description = self.func.__doc__ or f"Function {self.name}"
        
        # If parameters not provided, generate from function signature
        if not self.parameters:
            self.parameters = self._parameters_from_function()
        
        # Call parent validation
        super().__post_init__()
    
    def _parameters_from_function(self) -> List[ToolParameter]:
        """
        Generate parameters from the function signature.
        
        Returns:
            List[ToolParameter]: List of parameters for the tool.
        """
        parameters = []
        signature = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        docstring = inspect.getdoc(self.func) or ""
        
        # Parse docstring to extract parameter descriptions
        param_descriptions = {}
        current_param = None
        for line in docstring.split("\n"):
            line = line.strip()
            if line.startswith(":param "):
                parts = line[7:].split(":", 1)
                if len(parts) > 1:
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    param_descriptions[param_name] = param_desc
                    current_param = param_name
            elif current_param and line and not line.startswith(":"):
                param_descriptions[current_param] += " " + line
        
        # Create parameter definitions
        for name, param in signature.parameters.items():
            # Skip self parameter for methods
            if name == "self" and len(signature.parameters) > 0:
                continue
                
            param_type = type_hints.get(name, Any)
            type_str = self._python_type_to_json_schema_type(param_type)
            
            parameter = {
                "name": name,
                "description": param_descriptions.get(name, f"Parameter {name}"),
                "type": type_str,
                "required": param.default == inspect.Parameter.empty
            }
            
            parameters.append(parameter)
        
        return parameters
    
    def _python_type_to_json_schema_type(self, python_type: Type) -> str:
        """
        Convert Python type to JSON Schema type.
        
        Args:
            python_type: The Python type to convert.
            
        Returns:
            str: The corresponding JSON Schema type.
        """
        if python_type in (str, type(None)):
            return "string"
        elif python_type in (int, float):
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type in (list, tuple, set):
            return "array"
        elif python_type in (dict, Dict):
            return "object"
        else:
            return "string"  # Default to string for complex types
    
    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the wrapped function with the given parameters.
        
        Args:
            **kwargs: Parameters for the function.
            
        Returns:
            Any: The result of executing the function.
        """
        return self.func(**kwargs)