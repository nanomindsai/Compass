"""In-memory document store for Compass."""

from typing import Dict, List, Optional, Union, Any
import numpy as np

from compass.dataclasses import Document


class InMemoryDocumentStore:
    """
    In-memory document store for Compass.
    
    A simple document store that keeps documents and their embeddings in memory.
    Suitable for small datasets and prototyping.
    """
    
    def __init__(self) -> None:
        """Initialize an empty document store."""
        self.docs: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def write_documents(
        self, documents: List[Document], embeddings: Optional[List[np.ndarray]] = None
    ) -> List[str]:
        """
        Write documents to the store with optional embeddings.
        
        Args:
            documents: List of documents to write.
            embeddings: Optional list of embeddings corresponding to the documents.
            
        Returns:
            List[str]: IDs of the written documents.
        """
        doc_ids = []
        
        for i, doc in enumerate(documents):
            self.docs[doc.id] = doc
            doc_ids.append(doc.id)
            
            if embeddings and i < len(embeddings):
                self.embeddings[doc.id] = embeddings[i]
            
        return doc_ids
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: ID of the document to retrieve.
            
        Returns:
            Optional[Document]: The document if found, None otherwise.
        """
        return self.docs.get(doc_id)
    
    def get_documents(self, ids: Optional[List[str]] = None) -> List[Document]:
        """
        Retrieve multiple documents.
        
        Args:
            ids: Optional list of document IDs to retrieve. If None, retrieves all documents.
            
        Returns:
            List[Document]: List of retrieved documents.
        """
        if ids is None:
            return list(self.docs.values())
        
        return [self.docs[doc_id] for doc_id in ids if doc_id in self.docs]
    
    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Retrieve the embedding for a document.
        
        Args:
            doc_id: ID of the document.
            
        Returns:
            Optional[np.ndarray]: The embedding if found, None otherwise.
        """
        return self.embeddings.get(doc_id)
    
    def query_by_embedding(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Document]:
        """
        Find documents similar to the query embedding.
        
        Args:
            query_embedding: Embedding vector to compare against.
            top_k: Maximum number of documents to return.
            
        Returns:
            List[Document]: List of documents sorted by similarity.
        """
        if not self.embeddings:
            return []
        
        # Calculate cosine similarity
        scores = {}
        for doc_id, emb in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, emb)
            scores[doc_id] = similarity
        
        # Sort by similarity score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        
        # Retrieve documents
        result_docs = []
        for doc_id in sorted_ids:
            doc = self.docs.get(doc_id)
            if doc:
                doc_copy = Document(
                    content=doc.content,
                    meta=doc.meta.copy(),
                    id=doc.id,
                    score=scores[doc_id]
                )
                result_docs.append(doc_copy)
        
        return result_docs
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec_a: First vector.
            vec_b: Second vector.
            
        Returns:
            float: Cosine similarity (between -1 and 1).
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)
    
    def count_documents(self) -> int:
        """
        Count the number of documents in the store.
        
        Returns:
            int: Number of documents.
        """
        return len(self.docs)
    
    def delete_documents(self, ids: Optional[List[str]] = None) -> None:
        """
        Delete documents from the store.
        
        Args:
            ids: List of document IDs to delete. If None, deletes all documents.
        """
        if ids is None:
            self.docs.clear()
            self.embeddings.clear()
            return
        
        for doc_id in ids:
            if doc_id in self.docs:
                del self.docs[doc_id]
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]