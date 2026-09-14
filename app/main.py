import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.schemas import ChatRequest, ChatResponse, ToolCall
from app.agent import run_agent
from app.tools.get_appointment import init_appointments_db

load_dotenv()

app = FastAPI(
    title = "Healthcare Support & Care Navigation Agent",
    description = (
        "An agentic AI system that autonomously selects tools to answer "
        "healthcare queries — powered by Neo4j Knowledge Graph, SQLite, "
        "and OpenAI function calling."
    ),
    version = "1.0.0"
)

# Allow all origins for local dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ---------------------------------------------------------------------------
# Startup event — initialise SQLite appointments DB
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    print("[Startup] Initialising appointments database...")
    init_appointments_db()
    print("[Startup] Appointments DB ready.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status":  "ok",
        "service": "healthcare-support-agent",
        "version": "1.0.0"
    }

# ---------------------------------------------------------------------------
# Chat endpoint — the only endpoint the client needs
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Send a natural language healthcare query to the agent.

    The agent autonomously decides which tools to call (find_doctor,
    get_appointment, calculate_bmi), executes them, and returns a
    synthesised answer along with observability metadata.

    **Example queries:**
    - "Which doctors treat hypertension?"
    - "What is the status of appointment AP101?"
    - "My weight is 72 kg and height is 175 cm. What is my BMI?"
    - "Find me a cardiologist and calculate my BMI for 80 kg and 170 cm."
    """
    try:
        result = run_agent(request.message)

        return ChatResponse(
            answer      = result["answer"],
            tools_used  = [ToolCall(**t) for t in result["tools_used"]],
            total_turns = result["total_turns"]
        )

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Agent error: {str(e)}"
        )