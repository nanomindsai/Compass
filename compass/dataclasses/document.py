"""Document dataclass for Compass."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import uuid


@dataclass
class Document:
    """
    Base document class for Compass.
    
    Represents a document or a chunk of a document with associated metadata.
    """
    
    content: str
    """The text content of the document."""
    
    meta: Dict[str, Any] = field(default_factory=dict)
    """Metadata associated with the document."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for the document."""
    
    score: Optional[float] = None
    """Score assigned to the document (e.g., by a retriever or ranker)."""
    
    embedding: Optional[Union[list, "numpy.ndarray"]] = None
    """Document embedding if already computed."""
    
    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if not self.content and not self.meta:
            raise ValueError("Document must have either content or meta.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary."""
        doc_dict = {
            "content": self.content,
            "meta": self.meta,
            "id": self.id,
        }
        if self.score is not None:
            doc_dict["score"] = self.score
            
        # Don't include embedding in dict representation
        return doc_dict