"""Sentence Transformer embedder for Compass."""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from compass.pipeline import Component
from compass.dataclasses import Document


class SentenceTransformerEmbedder(Component):
    """
    A component for generating embeddings using the sentence-transformers library.
    
    Generates vector embeddings for text or documents using pre-trained models
    from the sentence-transformers library.
    """
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
    ):
        """
        Initialize the SentenceTransformerEmbedder.
        
        Args:
            model: Name of the sentence-transformers model to use.
            device: Device to use for computation (e.g., 'cpu', 'cuda', 'mps'). 
                    If None, uses the default device.
            prefix: Text to prepend to each document or text before embedding.
            suffix: Text to append to each document or text before embedding.
            batch_size: Batch size for processing multiple documents.
        """
        self.model_name = model
        self.device = device
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.model = None
    
    def _init_model(self) -> None:
        """Initialize the model if it hasn't been initialized yet."""
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The sentence-transformers package is not installed. "
                "Please install it with `pip install sentence-transformers`."
            )
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a list of texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List[np.ndarray]: List of embedding vectors.
        """
        self._init_model()
        
        # Apply prefix and suffix
        processed_texts = [f"{self.prefix}{text}{self.suffix}" for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            processed_texts, 
            batch_size=self.batch_size, 
            convert_to_numpy=True
        )
        
        return list(embeddings)
    
    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of documents to embed.
            
        Returns:
            List[np.ndarray]: List of embedding vectors.
        """
        texts = [doc.content for doc in documents]
        return self.embed_texts(texts)
    
    def run(
        self, documents: Optional[List[Document]] = None, text: Optional[str] = None, texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the embedder component.
        
        Args:
            documents: Optional list of documents to embed.
            text: Optional single text to embed.
            texts: Optional list of texts to embed.
            
        Returns:
            Dict containing the embeddings:
            - "document_embeddings": List of document embeddings if documents were provided
            - "text_embedding": Single embedding if text was provided
            - "text_embeddings": List of embeddings if texts were provided
            
        Raises:
            ValueError: If none of documents, text, or texts is provided.
        """
        if documents is None and text is None and texts is None:
            raise ValueError("At least one of 'documents', 'text', or 'texts' must be provided")
        
        result = {}
        
        if documents is not None:
            document_embeddings = self.embed_documents(documents)
            result["document_embeddings"] = document_embeddings
        
        if text is not None:
            text_embedding = self.embed_texts([text])[0]
            result["text_embedding"] = text_embedding
        
        if texts is not None:
            text_embeddings = self.embed_texts(texts)
            result["text_embeddings"] = text_embeddings
        
        return result