"""
Pydantic models for validating FastAPI request and response
payloads and defining the API contract.
"""

from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    """
    Incoming request body for POST /chat.

    'message' is the raw natural-language query from the user.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language healthcare query from the user.",
        examples=["Which doctors treat hypertension?"],
    )


class ToolCall(BaseModel):
    """
    Records one tool invocation that occurred during the agent loop.
    Returned in the response for observability and interview demonstration.
    """
    tool: str = Field(..., description="Name of the tool that was called.")
    args: dict[str, Any] = Field(..., description="Arguments passed to the tool.")


class ChatResponse(BaseModel):
    """
    Response body returned by POST /chat.

    Fields:
        answer      — LLM's final synthesised response.
        tools_used  — ordered list of tools called with their arguments.
        total_turns — number of tool round-trips (0 means direct answer).
    """
    answer: str = Field(..., description="LLM's final response to the user.")
    tools_used: list[ToolCall] = Field(
        default_factory=list,
        description="Tools called during this request, in execution order.",
    )
    total_turns: int = Field(
        ...,
        description="Number of agent→tool→agent round-trips. 0 = direct response.",
    )