"""
Agent example using Compass with official agent frameworks.

This example demonstrates how to use Compass with OpenAI's Agent SDK
and Anthropic's MCP (Model Context Protocol) framework.
"""

import os
import requests
import json
from datetime import datetime

from compass.components.tools import ToolRegistry, FunctionTool
from compass.components.agents import OpenAIAgent, AnthropicAgent


# Set up tool registry
tool_registry = ToolRegistry()

# Define some useful tools
@tool_registry.register_function
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about a topic.
    
    :param query: The search query
    :return: Wikipedia search results
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "srlimit": 3
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "query" in data and "search" in data["query"]:
        results = data["query"]["search"]
        formatted_results = []
        
        for result in results:
            title = result["title"]
            snippet = result["snippet"].replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            formatted_results.append(f"Title: {title}\nSnippet: {snippet}\n")
        
        return "\n".join(formatted_results)
    else:
        return "No results found"


@tool_registry.register_function
def get_current_weather(location: str) -> str:
    """
    Get the current weather for a location.
    
    :param location: The location to get weather for
    :return: Current weather information
    """
    # In a real application, you would call a weather API here
    # This is a mock implementation
    return f"Weather for {location}: 72°F, Partly Cloudy"


@tool_registry.register_function
def get_current_date() -> str:
    """
    Get the current date and time.
    
    :return: Current date and time
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


# Create the answer tool
def answer(answer: str) -> str:
    """
    Provide a final answer to the user.
    
    :param answer: The answer to provide
    :return: The answer
    """
    return answer

tool_registry.register_function(answer)


# Example with OpenAI Agent SDK
def run_openai_agent_sdk():
    """Run an example with the OpenAI Agent SDK."""
    print("=== OpenAI Agent SDK Example ===")
    
    # Make sure to set OPENAI_API_KEY environment variable or pass it directly
    agent = OpenAIAgent(
        tool_registry=tool_registry,
        model="gpt-4o",
        system_prompt="You are a helpful assistant. Use tools to provide accurate information."
    )
    
    # Run the agent with a query
    query = "What's the weather like in San Francisco and what day of the week is it today?"
    print(f"Query: {query}")
    
    result = agent.run(query)
    print("\nAnswer:")
    print(result["answer"])
    
    print("\nIntermediate steps:")
    for i, step in enumerate(result["intermediate_steps"]):
        print(f"Step {i+1}: Used tool '{step.action.tool_name}' with input: {step.action.tool_input}")
        print(f"Result: {step.observation}")
        print()
    
    # You can continue the conversation using the same thread
    thread_id = result["thread_id"]
    follow_up_query = "And what about the weather in New York?"
    print(f"\nFollow-up Query: {follow_up_query}")
    
    follow_up_result = agent.run(follow_up_query, thread_id=thread_id)
    print("\nFollow-up Answer:")
    print(follow_up_result["answer"])


# Example with Anthropic MCP
def run_anthropic_mcp():
    """Run an example with Anthropic's Model Context Protocol framework."""
    print("\n=== Anthropic MCP Example ===")
    
    # Make sure to set ANTHROPIC_API_KEY environment variable or pass it directly
    agent = AnthropicAgent(
        tool_registry=tool_registry,
        model="claude-3-opus-20240229",
        system_prompt="You are a helpful assistant. Use tools to provide accurate information."
    )
    
    # Run the agent with a query
    query = "What is the capital of France and what's the current date?"
    print(f"Query: {query}")
    
    result = agent.run(query)
    print("\nAnswer:")
    print(result["answer"])
    
    print("\nIntermediate steps:")
    for i, step in enumerate(result["intermediate_steps"]):
        print(f"Step {i+1}: Used tool '{step.action.tool_name}' with input: {step.action.tool_input}")
        print(f"Result: {step.observation}")
        print()


if __name__ == "__main__":
    # Uncomment to run the OpenAI example (requires API key)
    # run_openai_agent_sdk()
    
    # Uncomment to run the Anthropic example (requires API key)
    # run_anthropic_mcp()
    
    print("This example requires API keys to run.")
    print("Uncomment the run_openai_agent_sdk() or run_anthropic_mcp() calls after adding your API keys.")
    print("Set OPENAI_API_KEY environment variable for OpenAI or pass it directly.")
    print("Set ANTHROPIC_API_KEY environment variable for Anthropic or pass it directly.")