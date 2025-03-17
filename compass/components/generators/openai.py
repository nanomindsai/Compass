"""OpenAI generator implementation for Compass."""

from typing import Dict, List, Any, Optional, Union
import os

from compass.pipeline import Component
from compass.dataclasses import Document, Answer, ChatMessage


class OpenAIGenerator(Component):
    """
    A generator that uses OpenAI's API to generate text.
    
    Generates text using OpenAI models like GPT-3.5 and GPT-4.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        prompt_template: str = "Answer the following question based on the given context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
        max_tokens: int = 500,
        temperature: float = 0.7,
    ):
        """
        Initialize the OpenAIGenerator.
        
        Args:
            model: OpenAI model to use.
            api_key: OpenAI API key. If None, it will be read from the OPENAI_API_KEY environment variable.
            prompt_template: Template for constructing the prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Controls randomness in generation (0.0 to 1.0).
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Either pass it directly or set the OPENAI_API_KEY environment variable."
            )
        
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
    
    def _init_client(self) -> None:
        """Initialize the OpenAI client if it hasn't been initialized yet."""
        if self.client is not None:
            return
            
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The openai package is not installed. "
                "Please install it with `pip install openai`."
            )
        
        self.client = OpenAI(api_key=self.api_key)
    
    def generate(
        self, query: str, documents: Optional[List[Document]] = None, messages: Optional[List[ChatMessage]] = None
    ) -> Answer:
        """
        Generate an answer.
        
        Args:
            query: The query or question to answer.
            documents: Optional list of context documents.
            messages: Optional list of chat messages for chat-based generation.
            
        Returns:
            Answer: The generated answer.
        """
        self._init_client()
        
        if messages:
            # Chat-based generation
            openai_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer_text = response.choices[0].message.content.strip()
        else:
            # Prompt-based generation
            context = ""
            if documents:
                context = "\n\n".join([doc.content for doc in documents])
            
            prompt = self.prompt_template.format(context=context, query=query)
            
            if "gpt-3.5-turbo" in self.model or "gpt-4" in self.model:
                # These are chat models
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                answer_text = response.choices[0].message.content.strip()
            else:
                # These are completion models
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                answer_text = response.choices[0].text.strip()
        
        # Create the Answer object
        answer = Answer(
            answer=answer_text,
            documents=documents or [],
            meta={
                "model": self.model,
                "temperature": self.temperature,
            }
        )
        
        return answer
    
    def run(
        self,
        query: str,
        documents: Optional[List[Document]] = None,
        messages: Optional[List[ChatMessage]] = None,
    ) -> Dict[str, Any]:
        """
        Run the generator component.
        
        Args:
            query: The query or question to answer.
            documents: Optional list of context documents.
            messages: Optional list of chat messages for chat-based generation.
            
        Returns:
            Dict containing:
            - "answer": The generated Answer object
        """
        answer = self.generate(query=query, documents=documents, messages=messages)
        return {"answer": answer}