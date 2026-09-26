"""
app/main.py
-----------
FastAPI application entry point.
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
from app.schemas import ChatRequest, ChatResponse
from app.tools.get_appointment import init_appointments_db

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Healthcare Support Agent starting up.")
    try:
        settings.validate()
    except ValueError as e:
        log.error(f"Configuration error: {e}")
        raise

    init_appointments_db()
    log.info(
        f"Startup complete | model={settings.llm_model} | "
        f"neo4j={settings.neo4j_uri} | max_turns={settings.max_agent_turns}"
    )

    yield

    log.info("Shutting down.")
    close_neo4j_client()
    log.info("Shutdown complete.")


app = FastAPI(
    title="Healthcare Support & Care Navigation Agent",
    description=(
        "An agentic AI system that autonomously selects tools to answer "
        "healthcare queries — powered by a Neo4j Knowledge Graph, SQLite, "
        "and LangGraph orchestration with LLM function calling.\n\n"
        "**Disclaimer:** This system provides general informational support "
        "only. It does not provide medical diagnoses or clinical advice."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(f"Unhandled exception on {request.url.path} — {type(exc).__name__}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


@app.get("/", tags=["Health"])
def root_health_check() -> dict:
    return {"status": "ok", "service": "healthcare-support-agent", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def extended_health_check() -> dict:
    neo4j_ok = await asyncio.to_thread(_check_neo4j)
    status = "ok" if neo4j_ok else "degraded"
    return JSONResponse(
        status_code=200 if neo4j_ok else 503,
        content={
            "status": status,
            "components": {
                "neo4j": "ok" if neo4j_ok else "unavailable",
                "sqlite": "ok",
            },
        },
    )


def _check_neo4j() -> bool:
    try:
        return get_neo4j_client().verify_connectivity()
    except Exception:
        return False


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a natural language healthcare query to the agent.

    The agent decides which tools to call, executes them, and returns
    a synthesised answer.

    Example queries:
    - "Which doctors treat hypertension?"
    - "What is the status of appointment AP101?"
    - "My weight is 72 kg and height is 175 cm. What is my BMI?"
    """
    try:
        result = await asyncio.to_thread(run_agent, request.message)
        return ChatResponse(
            answer=result["answer"],
            total_turns=result["total_turns"],
        )
    except Exception as e:
        log.error(f"Agent error in /chat — {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="The agent encountered an error. Please try again.",
        )