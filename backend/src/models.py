"""
@file models.py
@description Pydantic data models matching TypeScript types from the frontend.
Defines the shape of API requests and responses for the RAG handbook system.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime


class HandbookDoc(BaseModel):
    """Represents a single handbook document with metadata."""
    id: str
    title: str
    category: str
    content: str


class SourceChunk(BaseModel):
    """A citation referencing a specific snippet from a handbook document."""
    docId: str
    snippet: str


class Message(BaseModel):
    """A single message in the chat history."""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    sources: Optional[List[SourceChunk]] = None
    highlights: Optional[Dict[str, List[str]]] = None
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""
    query: str
    history: List[Message] = []


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    content: str
    sources: List[SourceChunk]


class HighlightsRequest(BaseModel):
    """Request payload for the highlights endpoint."""
    answer: str
    document_content: str


class HighlightsResponse(BaseModel):
    """Response from the highlights endpoint."""
    highlights: List[str]
