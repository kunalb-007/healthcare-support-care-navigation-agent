import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from app.tools.find_doctor     import find_doctor
from app.tools.get_appointment import get_appointment
from app.tools.calculate_bmi  import calculate_bmi

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful, professional healthcare support assistant.

You help users:
- Find doctors and specialists for medical conditions
- Check appointment status and details
- Calculate and interpret BMI and weight classification

Rules:
1. ALWAYS use the find_doctor tool when asked about doctors, specialists,
   or which doctor treats a specific condition. Never guess doctor names
   from memory — the Knowledge Graph is the source of truth.
2. ALWAYS use get_appointment when the user provides an appointment ID.
3. ALWAYS use calculate_bmi when the user provides weight and height values.
4. You may answer general healthcare knowledge questions (e.g. "What is hypertension?")
   directly without a tool call.
5. Do not make up medical diagnoses or treatment recommendations.
6. Be clear, concise, and empathetic in your responses.
7. If a tool returns an error, inform the user clearly and suggest next steps.
"""

# ---------------------------------------------------------------------------
# Tool schemas — JSON Schema format; the description field is critical.
# The LLM reads descriptions to decide WHEN to call a tool.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "find_doctor",
            "description": (
                "Search the healthcare Knowledge Graph to find doctors and specialists. "
                "Use this tool when the user asks about: which doctor to see, "
                "which specialist treats a condition, finding a cardiologist / neurologist "
                "/ orthopedic doctor, or any doctor-search related query. "
                "Provide EITHER a medical condition (e.g. 'Hypertension', 'Migraine', "
                "'Knee Pain') OR a specialization (e.g. 'Cardiology', 'Neurology'). "
                "Do NOT provide both at the same time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type":        "string",
                        "description": (
                            "A medical condition or diagnosis. Examples: "
                            "'Hypertension', 'Heart Disease', 'Migraine', "
                            "'Knee Pain', 'Diabetes'."
                        )
                    },
                    "specialization": {
                        "type":        "string",
                        "description": (
                            "A medical specialization or department. Examples: "
                            "'Cardiology', 'Neurology', 'Orthopedics', 'General Medicine'."
                        )
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointment",
            "description": (
                "Look up an appointment by its unique appointment ID. "
                "Use this when the user mentions an appointment ID (e.g. 'AP101', 'AP102') "
                "and wants to know the status, doctor name, date, time, or any other "
                "appointment details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type":        "string",
                        "description": (
                            "The appointment ID provided by the user. "
                            "Examples: 'AP101', 'ap102', 'AP103'. "
                            "The tool normalises casing automatically."
                        )
                    }
                },
                "required": ["appointment_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": (
                "Calculate the Body Mass Index (BMI) for a person given their weight "
                "and height, and return the WHO weight classification category. "
                "Use this when the user provides their weight (in kg) and height (in cm) "
                "and asks about BMI, whether they are overweight, or their weight category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {
                        "type":        "number",
                        "description": "Body weight in kilograms. Must be a positive number."
                    },
                    "height_cm": {
                        "type":        "number",
                        "description": "Height in centimetres. Must be a positive number."
                    }
                },
                "required": ["weight_kg", "height_cm"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Tool registry — maps tool name to callable
# ---------------------------------------------------------------------------
TOOL_MAP = {
    "find_doctor":     find_doctor,
    "get_appointment": get_appointment,
    "calculate_bmi":   calculate_bmi,
}

MAX_TURNS = 10   # safety guard against infinite loops


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute the named tool with the provided arguments.
    Always returns a JSON string — never raises.
    Errors are captured and returned as JSON so the LLM can handle them.
    """
    if tool_name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})

    try:
        result = TOOL_MAP[tool_name](**tool_args)
        return json.dumps(result, ensure_ascii=False)
    except TypeError as e:
        # Wrong argument names / types from LLM
        return json.dumps({"error": f"Invalid arguments for tool '{tool_name}': {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Tool '{tool_name}' failed: {str(e)}"})


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agent(user_message: str) -> dict:
    """
    Run the full agent loop for a single user message.

    Flow:
      1. Send [system, user] to the LLM with tool schemas attached.
      2. If the LLM returns tool_calls:
           a. Execute each tool.
           b. Append the LLM message and each tool result to the history.
           c. Loop — send updated history back to the LLM.
      3. If the LLM returns a plain content message (no tool_calls):
           Return the answer along with observability metadata.

    Returns:
        {
            "answer":      str,          # LLM's final response
            "tools_used":  list[dict],   # [{tool, args}, ...]
            "total_turns": int           # number of tool round-trips
        }
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ]

    tools_called  = []
    turn_count    = 0

    while turn_count < MAX_TURNS:
        response = client.chat.completions.create(
            model       = "openai/gpt-4o-mini",
            messages    = messages,
            tools       = TOOL_SCHEMAS,
            tool_choice = "auto"          # LLM decides whether to call a tool
        )

        message = response.choices[0].message

        # ----------------------------------------------------------------
        # Case 1: LLM wants to call one or more tools
        # ----------------------------------------------------------------
        if message.tool_calls:
            # Add the LLM's assistant turn (including tool_calls) to history
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"[Agent][Turn {turn_count + 1}] Calling: {tool_name}({tool_args})")

                tool_result = dispatch_tool(tool_name, tool_args)

                print(f"[Agent][Turn {turn_count + 1}] Result : {tool_result[:200]}")

                tools_called.append({
                    "tool": tool_name,
                    "args": tool_args
                })

                # Add each tool result back into the conversation
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      tool_result
                })

            turn_count += 1
            # Loop — LLM will now see all tool results and decide next step

        # ----------------------------------------------------------------
        # Case 2: LLM produces a final answer (no tool calls)
        # ----------------------------------------------------------------
        else:
            final_answer = message.content or "I was unable to generate a response."
            return {
                "answer":      final_answer,
                "tools_used":  tools_called,
                "total_turns": turn_count
            }

    # If we exit the while loop, MAX_TURNS was hit
    return {
        "answer":      "The agent reached the maximum number of reasoning steps without "
                       "producing a final answer. Please rephrase your query.",
        "tools_used":  tools_called,
        "total_turns": turn_count
    }