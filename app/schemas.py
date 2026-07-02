"""Pydantic models matching the exact API schema required by the grading harness."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class Message(BaseModel):
    """A single message in the conversation history."""
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    """Incoming request to POST /chat — the full conversation so far."""
    messages: List[Message]


class Recommendation(BaseModel):
    """A single assessment recommendation — must be grounded in catalog data."""
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """
    Response from POST /chat.

    Hard constraints:
    - recommendations is [] (empty list, never null, never omitted) while clarifying/refusing
    - recommendations has 1-10 items inclusive when non-empty
    - end_of_conversation is a real boolean
    """
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status: str = "ok"
