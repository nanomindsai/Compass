"""Base Tool implementation for Compass."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union, TypedDict, Type
import inspect
import json
from abc import ABC, abstractmethod

from compass.pipeline import Component


class ToolParameter(TypedDict, total=False):
    """Definition of a tool parameter."""
    
    name: str
    description: str
    type: str
    required: bool
    enum: List[Any]


@dataclass
class Tool(Component, ABC):
    """
    Base class for all tools in Compass.
    
    A tool is a component that can be used by an agent or model to perform actions
    outside of its context. Tools can be functions, API calls, or other components.
    """
    
    name: str
    """The name of the tool."""
    
    description: str
    """A description of what the tool does."""
    
    parameters: List[ToolParameter] = field(default_factory=list)
    """The parameters for the tool."""
    
    def __post_init__(self) -> None:
        """Validate the tool after initialization."""
        if not self.name:
            raise ValueError("Tool must have a name")
        if not self.description:
            raise ValueError("Tool must have a description")
    
    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Parameters for the tool execution.
            
        Returns:
            Any: The result of executing the tool.
        """
        pass
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI's tool format.
        
        Returns:
            Dict[str, Any]: The tool in OpenAI's format.
        """
        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        for param in self.parameters:
            param_name = param["name"]
            parameters_schema["properties"][param_name] = {
                "type": param["type"],
                "description": param["description"]
            }
            
            if param.get("enum"):
                parameters_schema["properties"][param_name]["enum"] = param["enum"]
                
            if param.get("required", False):
                parameters_schema["required"].append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema
            }
        }
    
    def to_anthropic_tool(self) -> Dict[str, Any]:
        """
        Convert the tool to Anthropic's tool format.
        
        Returns:
            Dict[str, Any]: The tool in Anthropic's format.
        """
        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        for param in self.parameters:
            param_name = param["name"]
            parameters_schema["properties"][param_name] = {
                "type": param["type"],
                "description": param["description"]
            }
            
            if param.get("enum"):
                parameters_schema["properties"][param_name]["enum"] = param["enum"]
                
            if param.get("required", False):
                parameters_schema["required"].append(param_name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": parameters_schema
        }
    
    def run(self, tool_input: Optional[Union[str, Dict[str, Any]]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Run the tool component.
        
        Args:
            tool_input: The input to the tool, either as a JSON string or a dictionary.
            **kwargs: Additional parameters for the tool.
            
        Returns:
            Dict[str, Any]: Contains the tool result under the key 'result'.
        """
        # Parse the input if it's a string
        if isinstance(tool_input, str):
            try:
                params = json.loads(tool_input)
            except json.JSONDecodeError:
                params = {"input": tool_input}
        elif isinstance(tool_input, dict):
            params = tool_input
        else:
            params = kwargs
        
        # Execute the tool
        result = self.execute(**params)
        
        return {"result": result}