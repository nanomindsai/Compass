"""Vector retriever implementation for Compass."""

from typing import Dict, List, Any, Optional, Union
import numpy as np

from compass.pipeline import Component
from compass.dataclasses import Document
from compass.document_stores import InMemoryDocumentStore


class VectorRetriever(Component):
    """
    A retriever that uses vector similarity search.
    
    Retrieves documents from a document store based on the similarity between 
    the query embedding and document embeddings.
    """
    
    def __init__(
        self,
        document_store: InMemoryDocumentStore,
        top_k: int = 5,
    ):
        """
        Initialize the VectorRetriever.
        
        Args:
            document_store: The document store to retrieve from.
            top_k: Maximum number of documents to retrieve.
        """
        self.document_store = document_store
        self.top_k = top_k
    
    def retrieve(
        self, query_embedding: np.ndarray, top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve documents similar to the query embedding.
        
        Args:
            query_embedding: Embedding of the query.
            top_k: Optional override for the number of documents to retrieve.
            
        Returns:
            List[Document]: Retrieved documents sorted by relevance.
        """
        k = top_k if top_k is not None else self.top_k
        return self.document_store.query_by_embedding(query_embedding, top_k=k)
    
    def run(
        self, 
        text_embedding: Optional[np.ndarray] = None,
        query: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the retriever component.
        
        Args:
            text_embedding: Embedding of the query text.
            query: Raw query text (requires an embedder to be connected upstream).
            top_k: Optional override for the number of documents to retrieve.
            
        Returns:
            Dict containing:
            - "documents": List of retrieved documents
            
        Raises:
            ValueError: If neither text_embedding nor query is provided.
        """
        if text_embedding is None and query is None:
            raise ValueError("Either 'text_embedding' or 'query' must be provided")
        
        # If we're given a query but no embedding, raise an error
        if text_embedding is None and query is not None:
            raise ValueError(
                "Query text provided without an embedding. "
                "Connect an embedder component upstream to generate embeddings from queries."
            )
        
        k = top_k if top_k is not None else self.top_k
        documents = self.retrieve(text_embedding, top_k=k)
        
        return {"documents": documents}