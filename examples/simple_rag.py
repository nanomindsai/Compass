"""
Simple RAG example using Compass.

This example demonstrates how to create a basic RAG (Retrieval-Augmented Generation) 
system using Compass components.
"""

import os
from compass.dataclasses import Document
from compass.components.embedders import SentenceTransformerEmbedder
from compass.components.retrievers import VectorRetriever
from compass.components.generators import OpenAIGenerator
from compass.document_stores import InMemoryDocumentStore
from compass.pipeline import Pipeline

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"  # Replace with your actual API key

# Sample documents
documents = [
    Document(
        content=("Compass is a framework for building search and retrieval applications. "
                "It provides components for embedding, retrieval, and generation."),
        meta={"source": "documentation"}
    ),
    Document(
        content=("Compass is designed to be modular and extensible. Users can create "
                "custom components by inheriting from the Component base class."),
        meta={"source": "documentation"}
    ),
    Document(
        content=("RAG (Retrieval-Augmented Generation) is a technique that combines retrieval "
                "systems with generative models to produce more accurate and informed responses."),
        meta={"source": "article"}
    ),
    Document(
        content=("The Pipeline class in Compass allows you to connect components together "
                "and run them in sequence. It handles passing outputs from one component "
                "to the inputs of another based on the connections."),
        meta={"source": "tutorial"}
    ),
]

# Create components
document_store = InMemoryDocumentStore()
embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
retriever = VectorRetriever(document_store=document_store, top_k=2)
generator = OpenAIGenerator(
    model="gpt-3.5-turbo",
    prompt_template=(
        "Answer the following question based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )
)

# Create a pipeline
pipeline = Pipeline()
pipeline.add_component("embedder", embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("generator", generator)

# Connect components
pipeline.connect("embedder", "retriever")
pipeline.connect("retriever", "generator")

# Index documents
document_embeddings = embedder.embed_documents(documents)
document_store.write_documents(documents, document_embeddings)

# Query the pipeline
def query(question):
    """Query the pipeline with a question."""
    results = pipeline.run(query=question, text=question)
    
    # Get the answer
    answer = results["answer"]
    
    print(f"Question: {question}")
    print(f"Answer: {answer.answer}")
    print(f"Sources: {', '.join([doc.meta.get('source', 'unknown') for doc in answer.documents])}")
    print("-" * 50)

# Example queries
query("What is Compass?")
query("What is a RAG system?")
query("How do I connect components in Compass?")