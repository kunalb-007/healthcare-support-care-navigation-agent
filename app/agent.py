"""
LangGraph-based agent orchestration using a ReAct-style workflow.

The graph contains an agent node, a ToolNode, and conditional
routing based on LLM tool calls and the maximum turn limit.

The agent is stateless per request, and synchronous tool calls
execute sequentially.
"""

import json
import time
import uuid
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.config import settings
from app.logger import get_logger
from app.tools.calculate_bmi import calculate_bmi as _calculate_bmi
from app.tools.find_doctor import find_doctor as _find_doctor
from app.tools.get_appointment import get_appointment as _get_appointment

log = get_logger(__name__)

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful, professional healthcare support assistant.

You help users with:
- Finding doctors and specialists for medical conditions
- Checking appointment status and details
- Calculating and interpreting BMI

Rules:
1. ALWAYS use the find_doctor tool when asked about doctors, specialists,
   or which doctor treats a specific condition. Do not guess doctor names.
2. ALWAYS use get_appointment when the user provides an appointment ID.
3. ALWAYS use calculate_bmi when the user provides weight and height values.
4. You may answer general healthcare knowledge questions directly (e.g.
   "What is hypertension?") without calling a tool.
5. Do not make up medical diagnoses or treatment recommendations.
6. Be clear, concise, and empathetic.
7. If a tool returns an error, explain the situation clearly and suggest
   what the user can try next.

IMPORTANT: The BMI calculator and doctor search provide general information
only. They do not constitute medical advice or diagnosis. Always remind users
to consult a qualified healthcare professional for personalised guidance.
"""

# ------------------------------------------------------------------
# LangChain @tool wrappers
#
# LangGraph's ToolNode discovers tools from their function signature
# and docstring. The docstring is the critical part — the LLM reads
# it to decide WHEN to call each tool.
# ------------------------------------------------------------------

@lc_tool
def find_doctor(
        condition: str | None = None,
        specialization: str | None = None,
) -> dict:
    """
    Search the healthcare Knowledge Graph to find doctors and specialists.

    Use this tool when the user asks which doctor to see, which specialist
    treats a condition, or how to find a cardiologist, neurologist, or any
    other type of specialist.

    Provide EITHER:
      - condition: a medical condition e.g. 'Hypertension', 'Migraine',
        'Knee Pain', 'Diabetes'
      - specialization: a medical specialty e.g. 'Cardiology', 'Neurology',
        'Orthopedics', 'General Medicine'

    Do NOT provide both. If both are given, condition takes precedence.

    Returns a list of matching doctors with their hospital and experience.
    """
    return _find_doctor(condition=condition, specialization=specialization)


@lc_tool
def get_appointment(appointment_id: str) -> dict:
    """
    Look up an appointment by its unique appointment ID.

    Use this when the user mentions an appointment ID (e.g. 'AP101', 'AP102')
    and wants to know the status, doctor name, date, time, or any other
    appointment details. The ID is case-insensitive.

    Returns appointment details including patient name, doctor, date, time,
    status (confirmed/pending/cancelled), and any notes.
    """
    return _get_appointment(appointment_id=appointment_id)


@lc_tool
def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """
    Calculate Body Mass Index (BMI) given weight and height.

    Use this when the user provides their weight in kilograms and height
    in centimetres and asks about BMI, weight category, or whether they
    are underweight, overweight, or obese.

    Returns BMI value (rounded to 2 decimal places), WHO weight category,
    and brief general advice. This is NOT a medical diagnosis.
    """
    return _calculate_bmi(weight_kg=weight_kg, height_cm=height_cm)


# ------------------------------------------------------------------
# Registered tool list — explicit allowlist
#
# Only tools in this list can be called by the agent.
# Adding a function here makes it callable; removing it from this
# list makes it unreachable even if the LLM tries to invoke it.
#
# NOTE: This is an execution allowlist, not an authentication or
# authorization boundary. It does not replace access control.
# ------------------------------------------------------------------
REGISTERED_TOOLS = [find_doctor, get_appointment, calculate_bmi]

# ------------------------------------------------------------------
# Graph State
# ------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The full state passed between LangGraph nodes.

    Fields:
        messages:     Full conversation history. The add_messages
                      reducer appends new messages rather than replacing
                      the list, preserving the complete history.
        tools_used:   Ordered list of {tool, args} dicts for
                      observability — returned in the API response.
        total_turns:  Number of agent→tool→agent round-trips completed.
        request_id:   UUID assigned per /chat request for log correlation.
        error:        Set to a message string if a graph-level error occurs
                      (e.g. LLM unavailable). None during normal execution.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    tools_used: list[dict[str, Any]]
    total_turns: int
    request_id: str
    error: str | None


# ------------------------------------------------------------------
# LLM client
#
# ChatOpenAI from langchain_openai is used because it integrates
# directly with LangGraph's ToolNode via bind_tools().
# The base_url points to OpenRouter, which proxies to GPT-4o-mini.
#
# NOTE on timeouts: The underlying httpx client used by ChatOpenAI
# accepts a timeout parameter. We set request_timeout to 30 seconds.
# This is a best-effort timeout; network conditions may cause it to
# be exceeded in practice.
# ------------------------------------------------------------------
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        request_timeout=30,
        temperature=0,
    )


# ------------------------------------------------------------------
# Node: Agent
# ------------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    """
    LLM node: decides whether to call a tool or respond directly.

    Calls the LLM with the full message history and registered tools.
    The LLM's response is either:
      - An AIMessage with tool_calls → routed to tool_node
      - An AIMessage with plain text content → routed to END

    On LLM failure, sets state["error"] and returns a fallback message
    so the graph routes cleanly to END rather than crashing.
    """
    request_id = state.get("request_id", "unknown")
    turn = state["total_turns"] + 1

    log.info(
        "Agent node executing.",
        extra={"request_id": request_id, "turn": turn},
    )

    llm = _build_llm().bind_tools(REGISTERED_TOOLS)
    t0 = time.monotonic()

    try:
        response: AIMessage = llm.invoke(state["messages"])
        latency_ms = int((time.monotonic() - t0) * 1000)

        log.info(
            "LLM call complete.",
            extra={
                "request_id": request_id,
                "turn": turn,
                "has_tool_calls": bool(response.tool_calls),
                "latency_ms": latency_ms,
            },
        )
        return {"messages": [response]}

    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log.error(
            "LLM call failed.",
            extra={"request_id": request_id, "turn": turn, "error": str(e), "latency_ms": latency_ms},
        )
        # Return a safe fallback AIMessage so the graph can route to END.
        # Do NOT expose the raw exception to the user.
        fallback = AIMessage(
            content=(
                "I'm sorry, I'm unable to process your request right now due to a "
                "temporary service issue. Please try again in a moment."
            )
        )
        return {
            "messages": [fallback],
            "error": "LLM call failed. See server logs for details.",
        }


# ------------------------------------------------------------------
# Conditional routing
# ------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """
    Decide the next node after agent_node runs.

    Returns:
        "tools"  — if the last message has tool_calls and we're under MAX_TURNS
        END      — if no tool_calls, or MAX_TURNS reached, or an error occurred
    """
    # If a graph-level error occurred, stop.
    if state.get("error"):
        return END

    last_message = state["messages"][-1]

    # If the LLM produced tool calls and we haven't hit the turn limit
    if (
            isinstance(last_message, AIMessage)
            and last_message.tool_calls
            and state["total_turns"] < settings.max_agent_turns
    ):
        return "tools"

    # Log if we hit the iteration limit
    if state["total_turns"] >= settings.max_agent_turns:
        log.warning(
            "Max agent turns reached.",
            extra={
                "request_id": state.get("request_id", "unknown"),
                "max_turns": settings.max_agent_turns,
            },
        )

    return END


# ------------------------------------------------------------------
# Node: Tool execution
#
# LangGraph's built-in ToolNode handles:
#   - Reading tool_calls from the last AIMessage
#   - Calling each registered tool by name
#   - Catching errors per-tool and returning them as ToolMessages
#   - Appending ToolMessages to state.messages
#
# Unknown tool names produce a ToolMessage with an error string,
# which the LLM sees and reports to the user.
# ------------------------------------------------------------------
tool_node = ToolNode(REGISTERED_TOOLS)


# ------------------------------------------------------------------
# Tool tracking middleware
#
# LangGraph's ToolNode does not expose a hook to record which tools
# were called. We wrap the node to extract this from the messages.
# ------------------------------------------------------------------

def tool_node_with_tracking(state: AgentState) -> dict:
    """
    Execute tools via LangGraph's ToolNode and record what was called.

    Reads tool_calls from the last AIMessage, updates tools_used in
    state, then delegates to ToolNode for actual execution.
    """
    last_message = state["messages"][-1]
    new_tools_used = list(state.get("tools_used", []))

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            log.info(
                "Tool executing.",
                extra={
                    "request_id": state.get("request_id", "unknown"),
                    "tool": tool_name,
                    "tool_args": tool_args,
                    "turn": state["total_turns"] + 1,
                },
            )
            new_tools_used.append({"tool": tool_name, "args": tool_args})

    # Execute tools via ToolNode
    tool_result = tool_node.invoke(state)

    # Log tool results
    for msg in tool_result.get("messages", []):
        if isinstance(msg, ToolMessage):
            log.info(
                "Tool complete.",
                extra={
                    "request_id": state.get("request_id", "unknown"),
                    "tool_call_id": msg.tool_call_id,
                    "content_preview": str(msg.content)[:120],
                },
            )

    return {
        **tool_result,
        "tools_used": new_tools_used,
        "total_turns": state["total_turns"] + 1,
    }


# ------------------------------------------------------------------
# Graph compilation
# ------------------------------------------------------------------

def _build_graph() -> Any:
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_with_tracking)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    # After tool execution, always return to the agent node
    graph.add_edge("tools", "agent")

    return graph.compile()


# Compiled graph — built once at module load, reused per request
_graph = _build_graph()


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def run_agent(user_message: str) -> dict:
    """
    Run the LangGraph agent for a single user message.

    Initialises a fresh AgentState for each request (stateless — no
    memory across requests). The graph runs until the LLM produces
    a final answer or MAX_TURNS is reached.

    Args:
        user_message: Raw natural-language query from the user.

    Returns:
        dict with keys:
            answer      — LLM's final response string
            tools_used  — list of {tool, args} dicts
            total_turns — number of tool round-trips
    """
    request_id = str(uuid.uuid4())[:8]

    log.info(
        "Agent run started.",
        extra={"request_id": request_id, "message_preview": user_message[:80]},
    )
    t0 = time.monotonic()

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ],
        "tools_used": [],
        "total_turns": 0,
        "request_id": request_id,
        "error": None,
    }

    final_state = _graph.invoke(initial_state)

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Extract final answer from the last AIMessage in history
    answer = _extract_final_answer(final_state)

    log.info(
        "Agent run complete.",
        extra={
            "request_id": request_id,
            "total_turns": final_state["total_turns"],
            "tools_used": [t["tool"] for t in final_state["tools_used"]],
            "latency_ms": latency_ms,
        },
    )

    return {
        "answer": answer,
        "tools_used": final_state["tools_used"],
        "total_turns": final_state["total_turns"],
    }


def _extract_final_answer(state: AgentState) -> str:
    """
    Extract the last AIMessage content from state as the final answer.

    Searches backward through messages for the last AIMessage
    that has text content (not just tool calls).
    """
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)

    # Fallback — should not occur in normal operation
    if state.get("error"):
        return (
            "I'm sorry, I encountered an issue processing your request. "
            "Please try again."
        )
    return "I was unable to generate a response. Please try rephrasing your query."