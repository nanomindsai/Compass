"""OpenAI Agent SDK implementation for Compass."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import os

from compass.components.agents.base_agent import BaseAgent, AgentAction, AgentStep
from compass.dataclasses import ChatMessage
from compass.components.tools import ToolRegistry


@dataclass
class OpenAIAgent(BaseAgent):
    """
    An agent powered by OpenAI's Agent SDK for building agentic experiences.
    
    This agent uses OpenAI's official Agent SDK, which provides a framework
    for building complex, stateful agents with memory and tool use capabilities.
    
    See: https://platform.openai.com/docs/guides/agents-sdk
    """
    
    model: str = "gpt-4o"
    """The OpenAI model to use."""
    
    api_key: Optional[str] = None
    """OpenAI API key. If None, read from environment variable."""
    
    temperature: float = 0.7
    """Temperature for generation."""
    
    system_prompt: str = "You are a helpful assistant that can use tools to solve tasks."
    """System prompt providing context to the model."""
    
    max_agent_steps: int = 10
    """Maximum number of steps the agent can take."""
    
    client = None
    """OpenAI client (initialized lazily)."""
    
    agent = None
    """OpenAI Agent instance (initialized lazily)."""
    
    def __post_init__(self) -> None:
        """Initialize the agent."""
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Either pass it directly or set the OPENAI_API_KEY environment variable."
            )
    
    def _init_client(self) -> None:
        """Initialize the OpenAI client and Agent if not already initialized."""
        if self.client is not None:
            return
        
        try:
            from openai import OpenAI
            from openai.agents import Agent
            from openai.types.beta.thread import Thread
        except ImportError:
            raise ImportError(
                "The openai package is not installed or doesn't include Agent SDK. "
                "Please install it with `pip install openai>=1.27.0`."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Convert tools to OpenAI function format
        tools = self.tool_registry.to_openai_tools()
        
        # Create agent definition
        self.agent = Agent(
            client=self.client,
            model=self.model,
            instructions=self.system_prompt,
            tools=tools,
        )
    
    def _convert_openai_tool_calls_to_actions(
        self, 
        tool_calls: List[Any]
    ) -> List[AgentAction]:
        """
        Convert OpenAI Agent SDK tool calls to Compass AgentAction objects.
        
        Args:
            tool_calls: List of tool calls from OpenAI Agent SDK.
            
        Returns:
            List[AgentAction]: List of Compass AgentAction objects.
        """
        actions = []
        
        for call in tool_calls:
            actions.append(AgentAction(
                tool_name=call.function.name,
                tool_input=json.loads(call.function.arguments),
                thought=f"Using tool: {call.function.name}"
            ))
        
        return actions
    
    def _create_or_continue_thread(
        self, 
        query: str, 
        chat_history: Optional[List[ChatMessage]] = None,
        thread_id: Optional[str] = None
    ) -> Tuple[Any, str]:
        """
        Create a new thread or continue an existing one.
        
        Args:
            query: The query to process.
            chat_history: Optional chat history.
            thread_id: Optional thread ID to continue.
            
        Returns:
            Tuple[Any, str]: The thread object and its ID.
        """
        from openai.types.beta.threads import ThreadMessage, MessageContentText
        
        # If thread_id is provided, get the existing thread
        if thread_id:
            thread = self.client.beta.threads.retrieve(thread_id)
        else:
            # Create a new thread
            thread = self.client.beta.threads.create()
            
            # Add chat history if provided
            if chat_history:
                for message in chat_history:
                    self.client.beta.threads.messages.create(
                        thread_id=thread.id,
                        role=message.role.value,
                        content=message.content
                    )
        
        # Add the current query
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )
        
        return thread, thread.id
    
    def _execute_openai_run(
        self, 
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Execute the agent run and process the steps.
        
        Args:
            thread_id: The thread ID to run the agent on.
            
        Returns:
            Dict[str, Any]: Result with answer and steps.
        """
        self.intermediate_steps = []
        
        # Submit the run
        run = self.agent.submit(thread_id=thread_id)
        
        # Process the run steps
        step_count = 0
        
        while run.status in ["queued", "in_progress", "requires_action"] and step_count < self.max_agent_steps:
            if run.status == "requires_action":
                # Agent needs to call tools
                tool_outputs = []
                
                if run.required_action and run.required_action.submit_tool_outputs:
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    
                    # Convert to Compass actions
                    actions = self._convert_openai_tool_calls_to_actions(tool_calls)
                    
                    # Execute each tool call
                    for i, action in enumerate(actions):
                        tool_call_id = tool_calls[i].id
                        observation = self._execute_tool(action)
                        
                        # Record the step
                        step = AgentStep(action=action, observation=observation)
                        self.intermediate_steps.append(step)
                        
                        # Add the result to tool outputs
                        tool_outputs.append({
                            "tool_call_id": tool_call_id,
                            "output": json.dumps(observation) if isinstance(observation, (dict, list)) else str(observation)
                        })
                
                # Submit the tool outputs back to the agent
                run = self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
            
            # Wait for the run to complete or require more actions
            if run.status in ["queued", "in_progress"]:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
            
            step_count += 1
        
        # Get the final answer
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=1
        )
        
        # Get the latest assistant message
        if messages.data and messages.data[0].role == "assistant":
            answer = messages.data[0].content[0].text.value
        else:
            answer = "I wasn't able to complete the task."
        
        return {
            "answer": answer,
            "intermediate_steps": self.intermediate_steps,
            "thread_id": thread_id,
            "run_id": run.id
        }
    
    def plan(self, query: str, chat_history: Optional[List[ChatMessage]] = None) -> AgentAction:
        """
        Plan the next action to take based on the query and chat history.
        
        NOTE: For OpenAI Agent SDK, this is a placeholder as the actual planning
        happens in the run method. This is kept for compatibility with the
        BaseAgent interface.
        
        Args:
            query: The query or instruction for the agent.
            chat_history: Optional chat history for context.
            
        Returns:
            AgentAction: A placeholder action.
        """
        return AgentAction(
            tool_name="openai_agent_sdk",
            tool_input={"query": query},
            thought="Using OpenAI Agent SDK to process the query."
        )
    
    def process_step(self, action: AgentAction, observation: Any) -> Union[str, AgentAction]:
        """
        Process the result of an action and decide what to do next.
        
        NOTE: For OpenAI Agent SDK, this is a placeholder as the actual processing
        happens in the run method. This is kept for compatibility with the
        BaseAgent interface.
        
        Args:
            action: The action that was taken.
            observation: The result of the action.
            
        Returns:
            Union[str, AgentAction]: A placeholder result.
        """
        return "Using OpenAI Agent SDK to process the query."
    
    def run(
        self, 
        query: str, 
        chat_history: Optional[List[ChatMessage]] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the OpenAI Agent SDK agent to solve a task.
        
        Args:
            query: The query or instruction for the agent.
            chat_history: Optional chat history for context.
            thread_id: Optional thread ID to continue a conversation.
            
        Returns:
            Dict[str, Any]: Result containing the answer, intermediate steps, and thread/run IDs.
        """
        # Initialize the client and agent
        self._init_client()
        
        # Create or continue a thread
        _, thread_id = self._create_or_continue_thread(query, chat_history, thread_id)
        
        # Execute the run
        return self._execute_openai_run(thread_id)