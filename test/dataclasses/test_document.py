"""Tests for the Document dataclass."""

import pytest
from compass.dataclasses import Document


def test_document_initialization():
    """Test basic Document initialization."""
    # Test with just content
    doc = Document(content="Test content")
    assert doc.content == "Test content"
    assert doc.meta == {}
    assert doc.id is not None
    assert doc.score is None
    
    # Test with content and metadata
    doc = Document(content="Test with meta", meta={"source": "test"})
    assert doc.content == "Test with meta"
    assert doc.meta == {"source": "test"}
    
    # Test with custom ID
    doc = Document(content="Test with custom ID", id="custom-id")
    assert doc.id == "custom-id"
    
    # Test with score
    doc = Document(content="Test with score", score=0.95)
    assert doc.score == 0.95


def test_document_validation():
    """Test Document validation."""
    # Test that empty content and meta raises ValueError
    with pytest.raises(ValueError):
        Document(content="", meta={})


def test_document_to_dict():
    """Test Document.to_dict() method."""
    doc = Document(
        content="Test content",
        meta={"source": "test"},
        id="test-id",
        score=0.95
    )
    
    doc_dict = doc.to_dict()
    assert doc_dict["content"] == "Test content"
    assert doc_dict["meta"] == {"source": "test"}
    assert doc_dict["id"] == "test-id"
    assert doc_dict["score"] == 0.95