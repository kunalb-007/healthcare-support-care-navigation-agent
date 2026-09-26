"""
app/schemas.py
--------------
Pydantic models for FastAPI request and response bodies.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming request body for POST /chat.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language healthcare query from the user.",
        examples=["Which doctors treat hypertension?"],
    )


class ChatResponse(BaseModel):
    """
    Response body returned by POST /chat.

    Fields:
        answer      — LLM's final synthesised response.
        total_turns — number of tool round-trips (0 = direct answer).
    """
    answer: str = Field(..., description="LLM's final response to the user.")
    total_turns: int = Field(
        ...,
        description="Number of agent→tool→agent round-trips. 0 = direct response.",
    )