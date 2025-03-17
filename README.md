# Compass

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/compass/)

**Compass** is a flexible, modular framework for building search, retrieval, and knowledge applications powered by LLMs.

## Features

- 🧩 **Modular Components**: Mix and match embedders, retrievers, generators, and other components
- 🔗 **Easy Integration**: Simple interfaces for connecting with various LLM providers and vector databases
- 📊 **Evaluation Tools**: Built-in metrics and evaluation pipelines to measure performance
- 🚀 **Scalable**: From prototypes to production-ready applications
- 🔍 **Extensible**: Easily create custom components to suit your specific use case

## Installation

```bash
pip install compass
```

Or with specific features:

```bash
pip install compass[embedders,generators]
```

## Quick Start

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

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache License 2.0
