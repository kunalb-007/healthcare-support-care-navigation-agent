from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """
    Incoming request body for POST /chat.
    `message` is the raw natural-language query from the user.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language query from the user.",
        examples=["Which doctors treat hypertension?"]
    )


class ToolCall(BaseModel):
    """
    Records one tool invocation that occurred during the agent loop.
    Returned in the response for observability / debugging.
    """
    tool: str
    args: dict


class ChatResponse(BaseModel):
    """
    Response body returned by POST /chat.

    answer      — the LLM's final synthesised response.
    tools_used  — list of tools called and the arguments used.
    total_turns — how many tool round-trips occurred (0 = direct answer).
    """
    answer:  str
    tools_used:  List[ToolCall]
    total_turns: int