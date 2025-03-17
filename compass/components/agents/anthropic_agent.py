"""Anthropic MCP (Model Context Protocol) implementation for Compass."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import json
import os
import uuid

from compass.components.agents.base_agent import BaseAgent, AgentAction, AgentStep
from compass.dataclasses import ChatMessage
from compass.components.tools import ToolRegistry, Tool


@dataclass
class AnthropicAgent(BaseAgent):
    """
    An agent powered by Anthropic's Claude models with MCP (Model Context Protocol).
    
    This agent leverages Anthropic's MCP framework, which provides a structured protocol
    for Claude models to use tools and reason through complex tasks.
    
    See: https://docs.anthropic.com/en/docs/agents-and-tools/mcp
    """
    
    model: str = "claude-3-opus-20240229"
    """The Anthropic model to use."""
    
    api_key: Optional[str] = None
    """Anthropic API key. If None, read from environment variable."""
    
    temperature: float = 0.7
    """Temperature for generation."""
    
    system_prompt: str = "You are a helpful assistant that can use tools to solve tasks."
    """System prompt providing context to the model."""
    
    max_tokens: int = 4096
    """Maximum number of tokens in the response."""
    
    client = None
    """Anthropic client (initialized lazily)."""
    
    tool_map: Dict[str, Callable] = field(default_factory=dict)
    """Map of tool names to their implementation functions."""
    
    def __post_init__(self) -> None:
        """Initialize the agent."""
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Either pass it directly or set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Initialize tool_map from the registry
        for tool_name, tool in self.tool_registry.tools.items():
            self.tool_map[tool_name] = lambda **kwargs, tool=tool: tool.run(tool_input=kwargs)["result"]
    
    def _init_client(self) -> None:
        """Initialize the Anthropic client if not already initialized."""
        if self.client is not None:
            return
        
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is not installed. "
                "Please install it with `pip install anthropic>=0.18.0`."
            )
        
        self.client = Anthropic(api_key=self.api_key)
    
    def _format_messages(
        self, 
        query: str, 
        chat_history: Optional[List[ChatMessage]] = None
    ) -> List[Dict[str, Any]]:
        """
        Format messages for the Anthropic API.
        
        Args:
            query: The query for the agent.
            chat_history: Optional chat history.
            
        Returns:
            List[Dict[str, Any]]: Messages in Anthropic format.
        """
        messages = []
        
        # Add system prompt
        if self.system_prompt:
            messages.append({
                "role": "system", 
                "content": self.system_prompt
            })
        
        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                messages.append({
                    "role": message.role.value,
                    "content": message.content
                })
        
        # Add the current query
        messages.append({
            "role": "user", 
            "content": query
        })
        
        return messages
    
    def _prepare_tools_for_mcp(self) -> Dict[str, Any]:
        """
        Prepare tools in the Anthropic MCP format.
        
        Returns:
            Dict[str, Any]: Tools config for Anthropic MCP.
        """
        anthropic_tools = self.tool_registry.to_anthropic_tools()
        
        # Format tools for MCP
        tools_config = {}
        for tool in anthropic_tools:
            tool_name = tool["name"]
            tools_config[tool_name] = {
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            }
        
        return tools_config
    
    def _create_mcp_program(self) -> Dict[str, Any]:
        """
        Create a Model-Controlled Program config for Anthropic.
        
        Returns:
            Dict[str, Any]: MCP program configuration.
        """
        tools_config = self._prepare_tools_for_mcp()
        
        # Define the MCP program
        return {
            "tools": tools_config,
            "tool_choice": "auto"
        }
    
    def _execute_mcp_program(
        self, 
        messages: List[Dict[str, Any]],
        mcp_program: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the MCP program on the given messages.
        
        Args:
            messages: The messages to process.
            mcp_program: The MCP program configuration.
            
        Returns:
            Dict[str, Any]: Result with answer and steps.
        """
        from anthropic.types import MessageParam, ToolUseBlock
        
        self.intermediate_steps = []
        
        # Setup tool_map for MCP
        tool_map = {}
        for tool_name in mcp_program["tools"]:
            tool_map[tool_name] = self.tool_map[tool_name]
        
        # Initial message without tool use
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            tools=mcp_program["tools"],
            tool_choice=mcp_program["tool_choice"],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # Get all messages to track conversation
        all_messages = messages.copy()
        
        # Process tool use blocks if any
        step_count = 0
        max_steps = self.max_iterations
        
        while step_count < max_steps:
            # Check if response has tool use blocks
            tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
            
            if not tool_use_blocks:
                # No more tool uses, break the loop
                break
            
            # Add assistant response to messages
            all_messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Process each tool use
            user_message_blocks = []
            
            for tool_block in tool_use_blocks:
                # Convert to AgentAction for tracking
                action = AgentAction(
                    tool_name=tool_block.name,
                    tool_input=tool_block.input,
                    thought="Using Anthropic MCP to execute a tool."
                )
                
                # Execute the tool
                try:
                    tool_result = self.tool_map[tool_block.name](**tool_block.input)
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                
                # Record the step
                step = AgentStep(action=action, observation=tool_result)
                self.intermediate_steps.append(step)
                
                # Add tool result block
                user_message_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(tool_result) if isinstance(tool_result, (dict, list)) else str(tool_result)
                })
            
            # Add user message with tool results
            all_messages.append({
                "role": "user",
                "content": user_message_blocks
            })
            
            # Get next response
            response = self.client.messages.create(
                model=self.model,
                messages=all_messages,
                tools=mcp_program["tools"],
                tool_choice=mcp_program["tool_choice"],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            step_count += 1
        
        # Extract final answer from response
        text_blocks = [block.text for block in response.content if block.type == "text"]
        answer = "\n".join(text_blocks)
        
        return {
            "answer": answer,
            "intermediate_steps": self.intermediate_steps,
            "messages": all_messages
        }
    
    def plan(self, query: str, chat_history: Optional[List[ChatMessage]] = None) -> AgentAction:
        """
        Plan the next action to take based on the query and chat history.
        
        NOTE: For Anthropic MCP, this is a placeholder as the actual planning
        happens in the run method. This is kept for compatibility with the
        BaseAgent interface.
        
        Args:
            query: The query or instruction for the agent.
            chat_history: Optional chat history for context.
            
        Returns:
            AgentAction: A placeholder action.
        """
        return AgentAction(
            tool_name="anthropic_mcp",
            tool_input={"query": query},
            thought="Using Anthropic MCP to process the query."
        )
    
    def process_step(self, action: AgentAction, observation: Any) -> Union[str, AgentAction]:
        """
        Process the result of an action and decide what to do next.
        
        NOTE: For Anthropic MCP, this is a placeholder as the actual processing
        happens in the run method. This is kept for compatibility with the
        BaseAgent interface.
        
        Args:
            action: The action that was taken.
            observation: The result of the action.
            
        Returns:
            Union[str, AgentAction]: A placeholder result.
        """
        return "Using Anthropic MCP to process the query."
    
    def run(
        self, 
        query: str, 
        chat_history: Optional[List[ChatMessage]] = None
    ) -> Dict[str, Any]:
        """
        Run the Anthropic MCP agent to solve a task.
        
        Args:
            query: The query or instruction for the agent.
            chat_history: Optional chat history for context.
            
        Returns:
            Dict[str, Any]: Result containing the answer and intermediate steps.
        """
        # Initialize the client
        self._init_client()
        
        # Format the messages
        messages = self._format_messages(query, chat_history)
        
        # Create the MCP program
        mcp_program = self._create_mcp_program()
        
        # Execute the MCP program
        return self._execute_mcp_program(messages, mcp_program)