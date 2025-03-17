"""Base Agent implementation for Compass."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import json

from compass.pipeline import Component
from compass.dataclasses import ChatMessage
from compass.components.tools import ToolRegistry, Tool


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    
    tool_name: str
    """The name of the tool to use."""
    
    tool_input: Any
    """The input to provide to the tool."""
    
    thought: Optional[str] = None
    """The agent's reasoning for taking this action."""


@dataclass
class AgentStep:
    """Represents a step in an agent's execution."""
    
    action: AgentAction
    """The action taken by the agent."""
    
    observation: Any
    """The result of the action."""


@dataclass
class BaseAgent(Component, ABC):
    """
    Base class for all agents in Compass.
    
    An agent is a component that can use tools to solve tasks, making decisions
    about which tools to use and how to interpret their results.
    """
    
    tool_registry: ToolRegistry
    """Registry of tools available to the agent."""
    
    max_iterations: int = 10
    """Maximum number of iterations before the agent gives up."""
    
    intermediate_steps: List[AgentStep] = field(default_factory=list)
    """History of steps taken by the agent."""
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool for the agent to use.
        
        Args:
            tool: The tool to register.
        """
        self.tool_registry.register_tool(tool)
    
    def register_function(self, func: Optional[Callable] = None, *, 
                         name: Optional[str] = None, 
                         description: Optional[str] = None) -> Union[Tool, Callable]:
        """
        Register a function as a tool.
        
        Args:
            func: The function to register.
            name: Optional name for the tool.
            description: Optional description for the tool.
            
        Returns:
            Union[Tool, Callable]: The tool or a decorator.
        """
        return self.tool_registry.register_function(func, name=name, description=description)
    
    def _execute_tool(self, action: AgentAction) -> Any:
        """
        Execute a tool based on an agent action.
        
        Args:
            action: The action to execute.
            
        Returns:
            Any: The result of executing the tool.
            
        Raises:
            ValueError: If the tool doesn't exist.
        """
        tool = self.tool_registry.get_tool(action.tool_name)
        if not tool:
            raise ValueError(f"Tool '{action.tool_name}' not found")
            
        return tool.run(tool_input=action.tool_input)["result"]
    
    @abstractmethod
    def plan(self, query: str, chat_history: Optional[List[ChatMessage]] = None) -> AgentAction:
        """
        Plan the next action to take based on the query and chat history.
        
        Args:
            query: The query or instruction for the agent.
            chat_history: Optional chat history for context.
            
        Returns:
            AgentAction: The next action to take.
        """
        pass
    
    @abstractmethod
    def process_step(self, action: AgentAction, observation: Any) -> Union[str, AgentAction]:
        """
        Process the result of an action and decide what to do next.
        
        Args:
            action: The action that was taken.
            observation: The result of the action.
            
        Returns:
            Union[str, AgentAction]: Either a final answer or the next action to take.
        """
        pass
    
    def run(
        self, 
        query: str, 
        chat_history: Optional[List[ChatMessage]] = None
    ) -> Dict[str, Any]:
        """
        Run the agent to solve a task.
        
        Args:
            query: The query or instruction for the agent.
            chat_history: Optional chat history for context.
            
        Returns:
            Dict[str, Any]: Result containing the answer and intermediate steps.
        """
        self.intermediate_steps = []
        
        # Initial planning
        action = self.plan(query, chat_history)
        
        # Execute steps until we reach a conclusion or max iterations
        for _ in range(self.max_iterations):
            # Execute the action
            observation = self._execute_tool(action)
            
            # Record the step
            step = AgentStep(action=action, observation=observation)
            self.intermediate_steps.append(step)
            
            # Process the result
            result = self.process_step(action, observation)
            
            # If the result is a string, we're done
            if isinstance(result, str):
                return {
                    "answer": result,
                    "intermediate_steps": self.intermediate_steps
                }
            
            # Otherwise, we have a new action to execute
            action = result
        
        # If we've reached max iterations, return a timeout message
        return {
            "answer": "I've thought about this for too long without reaching a conclusion. Here's what I know so far: " + 
                     self._summarize_steps(),
            "intermediate_steps": self.intermediate_steps
        }
    
    def _summarize_steps(self) -> str:
        """
        Create a summary of intermediate steps.
        
        Returns:
            str: A summary of the steps taken.
        """
        summary = []
        for i, step in enumerate(self.intermediate_steps):
            summary.append(f"Step {i+1}:")
            if step.action.thought:
                summary.append(f"Thought: {step.action.thought}")
            summary.append(f"Action: {step.action.tool_name}")
            summary.append(f"Action Input: {step.action.tool_input}")
            summary.append(f"Observation: {step.observation}")
            summary.append("")
        
        return "\n".join(summary)