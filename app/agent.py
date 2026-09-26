"""
app/agent.py
------------
LangGraph-based agent orchestration.

Graph structure:
    User message
         |
    check_input_scope()   <-- guardrail (runs before graph)
         |
    [START] -> agent_node
                   |
             should_continue?
             /             \
         "tools"            END
            |
         tool_node
            |
         agent_node  (loop)

Guardrail:
    A simple scope check runs before the LangGraph graph is invoked.
    It rejects requests that are clearly outside healthcare navigation
    (e.g. asking for a diagnosis, prescription, or off-topic queries).
    This is NOT a sophisticated safety framework — it is a first-pass
    filter to keep the agent focused and to fail fast on unsupported requests.

State (AgentState):
    messages:    full conversation history using add_messages reducer
    total_turns: number of tool round-trips completed
    request_id:  short UUID for correlating log lines per request
    error:       set on LLM failure; causes routing to END immediately

Observability (what we log):
    - Request start with request_id
    - LLM call latency per turn
    - Tool name being executed
    - Errors with sanitized messages
    - Request completion with total_turns and total latency

Limitations:
    - Stateless: each /chat request starts a fresh graph with no
      memory of prior conversations.
    - Sequential tool execution: multiple tool calls in one LLM
      turn run one after another, not in parallel.
    - Guardrail is keyword/heuristic-based, not a trained classifier.
      Adversarial inputs may bypass it.
"""

import time
import uuid
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
#
# Explicitly restricts the agent to healthcare navigation.
# The LLM will refuse diagnosis/treatment questions based on this.
# Combined with the input guardrail for defence in depth.
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a healthcare navigation assistant. You help users with:
- Finding doctors and specialists for medical conditions
- Checking appointment status and details
- Calculating and interpreting BMI

You are NOT a doctor. You do NOT:
- Diagnose medical conditions
- Recommend specific treatments or medications
- Interpret medical test results
- Provide emergency medical advice

If a user asks for a diagnosis, treatment recommendation, or emergency help,
politely decline and direct them to consult a qualified healthcare professional
or call emergency services if urgent.

Rules:
1. Use find_doctor when asked about which doctor or specialist to see.
2. Use get_appointment when the user provides an appointment ID (e.g. AP101).
3. Use calculate_bmi when the user provides weight in kg and height in cm.
4. Answer general healthcare knowledge questions (e.g. "What is hypertension?")
   directly without a tool call.
5. If a tool returns an error, explain clearly and suggest next steps.

Always remind users that this system provides navigation assistance only —
not medical advice. Encourage consulting a qualified healthcare professional.
"""


# ------------------------------------------------------------------
# Guardrail
#
# Runs BEFORE the LangGraph graph is invoked.
# Purpose: fail fast on clearly out-of-scope or unsupported requests.
#
# This is intentionally simple — keyword matching plus length checks.
# It is not a trained safety classifier. Its job is to:
#   1. Reject requests asking for diagnosis or prescriptions
#   2. Reject requests that are clearly unrelated to healthcare
#   3. Reject suspiciously short or nonsensical inputs
#
# The system prompt provides a second layer inside the LLM itself.
# ------------------------------------------------------------------

# Keywords that suggest the user wants a diagnosis or treatment recommendation
_DIAGNOSIS_KEYWORDS = [
    "diagnose", "diagnosis", "prescribe", "prescription",
    "what disease do i have", "what is wrong with me",
    "do i have", "am i sick", "cure me", "treat me",
    "what medication", "which medicine", "what drug should i take",
]

# Keywords for requests clearly outside healthcare scope
_OFF_TOPIC_KEYWORDS = [
    "write code", "write an essay", "generate image", "play a game",
    "stock price", "weather", "recipe", "translate",
    "legal advice", "legal question",
]


def check_input_scope(message: str) -> str | None:
    """
    Basic scope check before passing the message to the agent.

    Returns:
        None   — if the message is acceptable, proceed to the graph.
        str    — a rejection reason string; run_agent returns this
                 directly without invoking the graph.

    Checks (in order):
        1. Minimum meaningful length (already enforced by Pydantic min_length=1,
           but we reject single-character or whitespace-only inputs here)
        2. Diagnosis / prescription requests
        3. Clearly off-topic requests
    """
    stripped = message.strip()

    # Reject trivially short inputs
    if len(stripped) < 3:
        return (
            "Your message is too short for me to understand. "
            "Please describe your healthcare question in more detail."
        )

    lowered = stripped.lower()

    # Reject diagnosis and prescription requests
    if any(kw in lowered for kw in _DIAGNOSIS_KEYWORDS):
        return (
            "I'm a healthcare navigation assistant — I can help you find the right "
            "doctor or specialist, but I cannot diagnose conditions or recommend "
            "medications. Please consult a qualified healthcare professional for "
            "diagnosis and treatment advice."
        )

    # Reject clearly off-topic requests
    if any(kw in lowered for kw in _OFF_TOPIC_KEYWORDS):
        return (
            "I'm a healthcare navigation assistant. I can help you find doctors, "
            "check appointment details, or calculate BMI. "
            "Your question appears to be outside that scope."
        )

    return None  # message is acceptable


# ------------------------------------------------------------------
# LangChain tool wrappers
#
# @lc_tool generates the JSON schema the LLM reads to decide when
# and how to call each tool. The docstring is the critical part —
# it is what the LLM reads as the tool's description.
# ------------------------------------------------------------------

@lc_tool
def find_doctor(
    condition: str | None = None,
    specialization: str | None = None,
) -> dict:
    """
    Search the healthcare Knowledge Graph to find doctors and specialists.

    Use when the user asks which doctor to see, which specialist treats
    a condition, or how to find a cardiologist, neurologist, etc.

    Provide EITHER:
      condition:      a medical condition e.g. 'Hypertension', 'Migraine'
      specialization: a medical specialty e.g. 'Cardiology', 'Neurology'

    Do NOT provide both. If both given, condition takes precedence.
    """
    return _find_doctor(condition=condition, specialization=specialization)


@lc_tool
def get_appointment(appointment_id: str) -> dict:
    """
    Look up an appointment by its unique ID (e.g. 'AP101', 'AP102').

    Use when the user provides an appointment ID and wants to know
    the status, doctor name, date, time, or any appointment details.
    The ID is case-insensitive.
    """
    return _get_appointment(appointment_id=appointment_id)


@lc_tool
def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """
    Calculate Body Mass Index (BMI) given weight in kg and height in cm.

    Use when the user provides weight and height and asks about BMI,
    their weight category, or whether they are overweight/underweight.

    Returns BMI value, WHO weight category, and brief general advice.
    This is NOT a medical diagnosis.
    """
    return _calculate_bmi(weight_kg=weight_kg, height_cm=height_cm)


# ------------------------------------------------------------------
# Tool registry — explicit execution allowlist
#
# Only tools in this list can be called by the agent.
# If the LLM hallucinates a tool name not in this list, ToolNode
# returns an error ToolMessage — the LLM sees it and reports to the
# user. No exception is raised.
#
# NOTE: This is an execution allowlist, not an authentication or
# authorization boundary.
# ------------------------------------------------------------------
REGISTERED_TOOLS = [find_doctor, get_appointment, calculate_bmi]


# ------------------------------------------------------------------
# Graph State
# ------------------------------------------------------------------

class AgentState(TypedDict):
    """
    State passed between LangGraph nodes on every step.

    messages:    Full conversation history. The add_messages reducer
                 APPENDS new messages rather than replacing the list —
                 this is how conversation context accumulates correctly
                 through multiple tool call cycles.
    total_turns: Number of agent→tool→agent round-trips completed.
                 Compared against MAX_AGENT_TURNS in should_continue().
    request_id:  Short UUID assigned per /chat request. Used to
                 correlate all log lines for a single request.
    error:       Set by agent_node on LLM failure. should_continue()
                 routes immediately to END when this is set.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    total_turns: int
    request_id: str
    error: str | None


# ------------------------------------------------------------------
# LLM client
#
# ChatOpenAI integrates with LangGraph's ToolNode via bind_tools().
# base_url points to GROQ which proxies to GPT-4o-mini.
# request_timeout=30 is a best-effort limit on the HTTP call.
# ------------------------------------------------------------------

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.groq_api_key,
        openai_api_base=settings.groq_base_url,
        request_timeout=30,
        temperature=0,
    )


# ------------------------------------------------------------------
# Node: Agent
# ------------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    """
    Calls the LLM with the current message history and bound tools.

    The LLM responds with either:
      - AIMessage with tool_calls  → should_continue routes to "tools"
      - AIMessage with plain text  → should_continue routes to END

    On LLM failure: sets state["error"] and returns a safe fallback
    message so the graph exits cleanly to END rather than crashing.
    """
    request_id = state["request_id"]
    turn = state["total_turns"] + 1

    llm = _build_llm().bind_tools(REGISTERED_TOOLS)
    t0 = time.monotonic()

    try:
        response: AIMessage = llm.invoke(state["messages"])
        latency_ms = int((time.monotonic() - t0) * 1000)
        log.info(f"[{request_id}] LLM turn {turn} completed in {latency_ms}ms")
        return {"messages": [response]}

    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log.error(f"[{request_id}] LLM call failed at turn {turn} after {latency_ms}ms — {type(e).__name__}")
        fallback = AIMessage(
            content=(
                "I'm sorry, I'm unable to process your request right now due to a "
                "temporary service issue. Please try again in a moment."
            )
        )
        return {
            "messages": [fallback],
            "error": f"LLM call failed: {type(e).__name__}",
        }


# ------------------------------------------------------------------
# Conditional routing
# ------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """
    Decides the next node after agent_node runs.

    Returns "tools" if the LLM produced tool calls and we are under
    the iteration limit. Returns END in all other cases.

    Routing rules (checked in order):
      1. state["error"] is set        → END  (LLM failure)
      2. total_turns >= MAX_TURNS     → END  (iteration limit hit)
      3. last message has tool_calls  → "tools"
      4. otherwise                    → END  (final answer)
    """
    if state.get("error"):
        return END

    if state["total_turns"] >= settings.max_agent_turns:
        log.warning(
            f"[{state['request_id']}] Max agent turns ({settings.max_agent_turns}) reached"
        )
        return END

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


# ------------------------------------------------------------------
# Node: Tool execution
#
# LangGraph's built-in ToolNode:
#   - Reads tool_calls from the last AIMessage
#   - Looks up each tool by name in REGISTERED_TOOLS
#   - Calls it with the provided arguments
#   - Formats the return value as a ToolMessage
#   - Appends ToolMessages to state.messages
#
# We add one wrapper layer solely to log which tool is executing.
# ------------------------------------------------------------------

_tool_node = ToolNode(REGISTERED_TOOLS)


def tool_node(state: AgentState) -> dict:
    """
    Logs tool execution then delegates to LangGraph's ToolNode.

    The only reason this wrapper exists is to log the tool name
    and increment total_turns. ToolNode handles all execution logic.
    """
    request_id = state["request_id"]
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tc in last_message.tool_calls:
            log.info(f"[{request_id}] Executing tool: {tc['name']} | args: {tc['args']}")

    result = _tool_node.invoke(state)

    return {
        **result,
        "total_turns": state["total_turns"] + 1,
    }


# ------------------------------------------------------------------
# Graph compilation
# ------------------------------------------------------------------

def _build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    # After tool execution always return to the agent node
    graph.add_edge("tools", "agent")

    return graph.compile()


# Compiled once at module load — reused for every request
_graph = _build_graph()


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def run_agent(user_message: str) -> dict:
    """
    Entry point called by the FastAPI /chat endpoint.

    Flow:
      1. Assign a request_id for log correlation
      2. Run check_input_scope() — reject out-of-scope requests immediately
      3. Invoke the LangGraph graph with initial state
      4. Extract and return the final answer

    Returns:
        answer:      LLM's final response string
        total_turns: number of tool round-trips that occurred
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.monotonic()

    log.info(f"[{request_id}] Request received")

    # -- Guardrail: scope check before touching the graph --
    rejection = check_input_scope(user_message)
    if rejection:
        log.info(f"[{request_id}] Request rejected by input guardrail")
        return {"answer": rejection, "total_turns": 0}

    # -- Build initial state --
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ],
        "total_turns": 0,
        "request_id": request_id,
        "error": None,
    }

    # -- Run the graph --
    final_state = _graph.invoke(initial_state)

    total_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        f"[{request_id}] Request complete | "
        f"turns={final_state['total_turns']} | "
        f"total_latency={total_ms}ms"
    )

    return {
        "answer": _extract_final_answer(final_state),
        "total_turns": final_state["total_turns"],
    }


def _extract_final_answer(state: AgentState) -> str:
    """
    Walk backwards through messages to find the last AIMessage with content.
    This is the LLM's final synthesised response.
    """
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)

    if state.get("error"):
        return (
            "I'm sorry, I encountered an issue processing your request. "
            "Please try again."
        )
    return "I was unable to generate a response. Please try rephrasing your query."