# Compass

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/compass/)

**Compass** is a flexible, modular framework for building search, retrieval, and knowledge navigation applications powered by LLMs and embeddings. It features official support for agent frameworks like OpenAI's Agent SDK and Anthropic's Model Context Protocol (MCP).

## Features

- 🧩 **Modular Components**: Mix and match embedders, retrievers, generators, and other components
- 🔗 **Easy Integration**: Simple interfaces for connecting with various LLM providers and vector databases
- 🤖 **Official Agent SDKs**: Built-in support for OpenAI's Assistants API (Agent SDK) and Anthropic's Model Context Protocol (MCP)
- 🛠️ **Tool Framework**: Create and manage tools for agents to use, with automatic schema generation
- 📊 **Evaluation Tools**: Built-in metrics and evaluation pipelines to measure performance
- 🚀 **Scalable**: From prototypes to production-ready applications
- 🔍 **Extensible**: Easily create custom components to suit your specific use case

## Installation

```bash
pip install compass
```

Or with specific features:

```bash
pip install compass[embedders,generators,agents]
```

## Quick Start: RAG Pipeline

```python
from compass.components.embedders import SentenceTransformerEmbedder
from compass.components.retrievers import VectorRetriever
from compass.components.generators import OpenAIGenerator
from compass.document_stores import InMemoryDocumentStore
from compass.pipeline import Pipeline
from compass.dataclasses import Document

# Create components
document_store = InMemoryDocumentStore()
embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
retriever = VectorRetriever(document_store=document_store)
generator = OpenAIGenerator(model="gpt-3.5-turbo")

# Create and index documents
documents = [
    Document(content="Compass is a framework for search and retrieval applications."),
    Document(content="Compass makes it easy to build RAG applications."),
]
embeddings = embedder.embed_documents(documents)
document_store.write_documents(documents, embeddings)

# Create a pipeline
pipe = Pipeline()
pipe.add_component("retriever", retriever)
pipe.add_component("generator", generator)
pipe.connect("retriever", "generator")

# Run the pipeline
results = pipe.run(query="What is Compass?")
print(results["generator"])
```

## Quick Start: OpenAI Agent SDK

```python
from compass.components.tools import ToolRegistry, FunctionTool
from compass.components.agents import OpenAIAgent

# Create a tool registry
tool_registry = ToolRegistry()

# Register a function as a tool
@tool_registry.register_function
def search_knowledge_base(query: str) -> str:
    """
    Search a knowledge base for information.
    
    :param query: The search query
    :return: Search results
    """
    # In a real application, you would search your knowledge base
    return f"Results for '{query}': Compass is a framework for building RAG applications."

# Create an agent using OpenAI's Agent SDK
agent = OpenAIAgent(
    tool_registry=tool_registry,
    model="gpt-4o",
    system_prompt="You are a helpful assistant. Use tools to provide accurate information."
)

# Run the agent
result = agent.run("Tell me about Compass framework")
print(result["answer"])

# Continue the conversation using the same thread
thread_id = result["thread_id"]
follow_up = agent.run("What can I build with it?", thread_id=thread_id)
print(follow_up["answer"])
```

## Quick Start: Anthropic MCP

```python
from compass.components.tools import ToolRegistry, FunctionTool
from compass.components.agents import AnthropicAgent

# Create a tool registry and register tools
# (This can be the same registry used for OpenAI)
tool_registry = ToolRegistry()
# ... register tools as in previous example

# Create an agent using Anthropic's Model Context Protocol
agent = AnthropicAgent(
    tool_registry=tool_registry,
    model="claude-3-opus-20240229",
    system_prompt="You are a helpful assistant. Use tools when necessary."
)

# Run the agent
result = agent.run("What can you tell me about France?")
print(result["answer"])
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache License 2.0