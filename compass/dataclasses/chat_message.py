"""Chat message dataclass for Compass."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


class Role(str, Enum):
    """Enumeration of possible roles in a chat conversation."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class ChatMessage:
    """
    A message in a chat conversation.
    
    Represents a single message in a chat conversation with a specified role and content.
    """
    
    role: Role
    """The role of the message sender (system, user, assistant, or function)."""
    
    content: str
    """The content of the message."""
    
    name: Optional[str] = None
    """Optional name of the sender, useful for function calls or multi-user chats."""
    
    meta: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata associated with the message."""
    
    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Create a user message."""
        return cls(role=Role.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the chat message to a dictionary."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.meta:
            result["meta"] = self.meta
        return result