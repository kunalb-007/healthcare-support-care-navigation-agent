"""
FastAPI application entry point.

Provides chat and health-check endpoints, manages application
lifespan, validates configuration, and runs synchronous agent
workloads in a background thread.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import run_agent
from app.config import settings
from app.graph.neo4j_client import close_neo4j_client, get_neo4j_client
from app.logger import get_logger
from app.schemas import ChatRequest, ChatResponse, ToolCall
from app.tools.get_appointment import init_appointments_db

log = get_logger(__name__)


# ------------------------------------------------------------------
# Lifespan — startup and shutdown logic
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Startup:
        1. Validate configuration (fails fast on missing secrets).
        2. Initialise the SQLite appointments database.
        3. Lazily connect to Neo4j (verified on /health, not here,
           so the app starts even if Neo4j is briefly unavailable).

    Shutdown:
        1. Close the Neo4j driver cleanly.
    """
    # ---- Startup ----
    log.info("Healthcare Support Agent starting up.")

    try:
        settings.validate()
    except ValueError as e:
        log.error(f"Configuration error: {e}")
        raise  # Prevent the app from starting with bad config

    log.info("Initialising appointments database.")
    init_appointments_db()

    log.info(
        "Startup complete.",
        extra={
            "neo4j_uri": settings.neo4j_uri,
            "llm_model": settings.llm_model,
            "max_agent_turns": settings.max_agent_turns,
            "allowed_origins": settings.allowed_origins,
        },
    )

    yield  # Application runs here

    # ---- Shutdown ----
    log.info("Healthcare Support Agent shutting down.")
    close_neo4j_client()
    log.info("Shutdown complete.")


# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

app = FastAPI(
    title="Healthcare Support & Care Navigation Agent",
    description=(
        "An agentic AI system that autonomously selects tools to answer "
        "healthcare queries — powered by a Neo4j Knowledge Graph, SQLite, "
        "and LangGraph orchestration with LLM function calling.\n\n"
        "**Disclaimer:** This system provides general informational support "
        "only. It does not provide medical diagnoses or clinical advice. "
        "Always consult a qualified healthcare professional."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# CORS middleware
#
# Origins are loaded from the ALLOWED_ORIGINS environment variable.
# Default is localhost:3000 (suitable for local dev only).
#
# IMPORTANT: Do not set ALLOWED_ORIGINS=* in production.
# In production, set it to your actual frontend domain(s).
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ------------------------------------------------------------------
# Global exception handler
#
# Catches any unhandled exception and returns a safe JSON response.
# Raw exception messages are logged but NOT returned to the client.
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(
        "Unhandled exception.",
        extra={"path": request.url.path, "error": str(exc)},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# ------------------------------------------------------------------
# Health check endpoints
# ------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root_health_check() -> dict:
    """Basic liveness check — returns immediately without DB queries."""
    return {
        "status": "ok",
        "service": "healthcare-support-agent",
        "version": "2.0.0",
    }


@app.get("/health", tags=["Health"])
async def extended_health_check() -> dict:
    """
    Extended health check — also verifies Neo4j connectivity.

    Returns:
        200 with status details if all components are reachable.
        503 if Neo4j is unavailable.

    NOTE: SQLite does not require a persistent connection, so it is
    not separately checked here.
    """
    neo4j_ok = await asyncio.to_thread(_check_neo4j)

    status = "ok" if neo4j_ok else "degraded"
    http_status = 200 if neo4j_ok else 503

    return JSONResponse(
        status_code=http_status,
        content={
            "status": status,
            "components": {
                "neo4j": "ok" if neo4j_ok else "unavailable",
                "sqlite": "ok",
            },
        },
    )


def _check_neo4j() -> bool:
    """Synchronous Neo4j connectivity check — run via to_thread."""
    try:
        return get_neo4j_client().verify_connectivity()
    except Exception:
        return False


# ------------------------------------------------------------------
# Chat endpoint
# ------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a natural language healthcare query to the agent.

    The agent autonomously decides which tools to call (find_doctor,
    get_appointment, calculate_bmi), executes them in sequence, and
    returns a synthesised answer along with observability metadata.

    **Example queries:**
    - "Which doctors treat hypertension?"
    - "What is the status of appointment AP101?"
    - "My weight is 72 kg and height is 175 cm. What is my BMI?"
    - "I have a migraine. Who should I see?"

    **Note:** This endpoint runs synchronous Neo4j and SQLite operations
    in a thread pool via asyncio.to_thread() to avoid blocking the event loop.
    """
    try:
        # run_agent is synchronous (Neo4j + SQLite + LLM calls).
        # asyncio.to_thread() runs it in a thread pool, preventing
        # the async event loop from being blocked.
        result = await asyncio.to_thread(run_agent, request.message)

        return ChatResponse(
            answer=result["answer"],
            tools_used=[ToolCall(**t) for t in result["tools_used"]],
            total_turns=result["total_turns"],
        )

    except Exception as e:
        log.exception(
            "Agent error in /chat endpoint.",
            extra={"error": str(e)},
        )

        raise HTTPException(
            status_code=500,
            detail="The agent encountered an error. Please check server logs.",
        ) from e