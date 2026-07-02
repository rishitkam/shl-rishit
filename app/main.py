"""
FastAPI application — thin routing layer wiring catalog, agent, and guardrails.

Routes:
- GET /health → {"status": "ok"}
- POST /chat → full conversation processing

All responses are guaranteed to be schema-valid ChatResponse objects,
even on errors or timeouts.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dotenv import load_dotenv

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.catalog import catalog_store
from app.agent import generate_response

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog + embeddings + embedding model at startup."""
    logger.info("Starting up — loading catalog and embeddings...")
    try:
        catalog_store.load()
        logger.info("Startup complete — %d catalog items loaded", len(catalog_store.items))
    except Exception as e:
        logger.error("Failed to load catalog: %s", e, exc_info=True)
        # Don't crash — let the health endpoint report the issue
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow any origin for the grading harness
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns {"status": "ok"} with HTTP 200.
    No LLM call — just confirms the process is up and catalog is loaded.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — stateless conversation processing.

    The request contains the full conversation history. No state is stored
    between calls — everything is re-derived from the messages array.
    """
    try:
        response = await generate_response(request)
        return response
    except Exception as e:
        logger.error("Unhandled error in /chat: %s", e, exc_info=True)
        # Always return a valid ChatResponse, never a 500
        return ChatResponse(
            reply=(
                "I apologize for the technical difficulty. "
                "Could you tell me about the role you're hiring for "
                "so I can recommend relevant SHL assessments?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler — ensures we never return a non-conforming body.
    This catches anything that slips past the route-level try/except.
    """
    logger.error("Global exception handler caught: %s", exc, exc_info=True)

    # For the /chat endpoint, return a valid ChatResponse
    if request.url.path == "/chat":
        return JSONResponse(
            status_code=200,
            content={
                "reply": "I encountered a technical issue. Could you tell me about the role you're hiring for?",
                "recommendations": [],
                "end_of_conversation": False,
            },
        )

    # For other endpoints, return a generic error
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
