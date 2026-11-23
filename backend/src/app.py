"""
FastAPI application for the RAG handbook backend.

Entry point for the backend. Exposes the REST API; on AWS the frontend is served by
CloudFront from S3, and locally (or in a plain container) it can be served from FRONTEND_PATH.

API Endpoints:
- GET  /api/health       — Health check and document count
- GET  /api/handbook     — All handbook documents (for source viewer)
- POST /api/contact      — Contact / feedback email
- POST /api/chat         — RAG chat (query + history → answer + sources)
- POST /api/chat/stream  — Same, as Server-Sent Events (tool steps live, then the answer)

Request flow:
  /api/chat → RAGService.get_rag_response() → rag.agent_pipeline

Environment variables (.env locally; SSM /nexus/ on AWS, see src/ssm.py):
- GROQ_API_KEY, OPENAI_API_KEY, CHROMA_API_KEY, TENANT_CHROMA, RESEND_API_KEY, CONTACT_EMAIL
- TABLE_NAME (AWS: chat logs + rate limits), ORIGIN_SECRET (AWS: only CloudFront may call)
- FRONTEND_PATH (optional, for serving a built frontend from this server)
"""

import asyncio
import hmac
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from utils.models import (
    ChatRequest,
    ChatResponse,
    ContactRequest,
    HandbookDoc,
    ToolStep,
)

from . import chat_logger, ratelimit, ssm
from .config_loader import load_config
from .contact_service import send_contact_email
from .handbook_loader import load_handbook_documents
from .rag_service import RAGService
from .settings import on_aws, origin_secret

# Load environment variables from .env or .env.local
load_dotenv()

# While the agent works, a comment line goes out this often so CloudFront (which drops a
# response that is silent for its origin read timeout) keeps the stream open.
HEARTBEAT_SECONDS = 15.0

# Global state: Loaded at startup and cached in memory
# These are read-only after initialization, so thread-safe
handbook_docs: list[HandbookDoc] = []
rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Initialize application state on server startup.

    1. On AWS, loads the secrets from SSM into the environment
    2. Loads all markdown handbook documents from data/handbook/
    3. Initializes the RAG service (Chroma Cloud connection + BM25 index)

    If any of this fails, the server fails to start.
    """
    global handbook_docs, rag_service
    try:
        ssm.load_parameters()

        # Load all handbook documents (used by /api/handbook endpoint)
        handbook_docs = load_handbook_documents()
        print(f"✓ Loaded {len(handbook_docs)} handbook documents")

        # Initialize RAG service with config (config.yaml in backend/)
        config = load_config()
        rag_service = RAGService(config=config, handbook_docs=handbook_docs)
    except Exception as e:
        print(f"✗ Error during startup: {e}")
        raise
    yield


app = FastAPI(
    title="RAG Company Handbook API",
    description="Backend API for the RAG Company Handbook chatbot",
    version="1.0.0",
    # Interactive API docs locally only; on AWS they would just map the API for strangers.
    docs_url=None if on_aws() else "/docs",
    redoc_url=None if on_aws() else "/redoc",
    openapi_url=None if on_aws() else "/openapi.json",
    lifespan=lifespan,
)

# CORS for running the Vite dev server against a backend on another port without its proxy.
# On AWS the frontend and API share one origin (CloudFront), so CORS never applies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_cloudfront(request: Request, call_next):
    """On AWS only CloudFront may call the API: it adds X-Origin-Verify, and it is what sets
    the viewer address header that rate limits trust."""
    expected = origin_secret()
    if expected and not hmac.compare_digest(
        request.headers.get("x-origin-verify", "").encode(), expected.encode()
    ):
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return await call_next(request)


async def _with_heartbeat(events, interval: float):
    """Yield the events of an async iterator, plus None whenever `interval` passes without one."""
    iterator = events.__aiter__()
    pending = asyncio.ensure_future(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield None
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            yield event
            pending = asyncio.ensure_future(iterator.__anext__())
    finally:
        pending.cancel()


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.

    Returns server status and the count of loaded handbook documents.
    Useful for monitoring, container orchestration, and debugging.

    Returns:
        dict: {"status": "healthy", "documents_loaded": <count>}
    """
    return {"status": "healthy", "documents_loaded": len(handbook_docs)}


@app.get("/api/handbook")
async def get_handbook():
    """
    Returns all handbook documents with metadata.

    The frontend uses this to:
    - Display available documents in the source viewer
    - Match document IDs from citations to full content
    - Show document titles and categories

    Returns:
        list[dict]: Array of HandbookDoc objects with id, title, category, content
    """
    return [doc.model_dump() for doc in handbook_docs]


@app.post("/api/contact", status_code=204)
async def contact(request: ContactRequest, http_request: Request):
    """
    Contact / feedback endpoint.

    Sends an email to the configured CONTACT_EMAIL address via Resend.
    Subject is prefixed with [Feedback] or [Get in Touch] depending on type.

    Returns 204 No Content on success.
    Raises:
        HTTPException 429: If this visitor sent too many messages today.
        HTTPException 500: If the email could not be sent.
    """
    if not await asyncio.to_thread(ratelimit.contact_allowed, http_request):
        raise HTTPException(status_code=429, detail=ratelimit.CONTACT_LIMIT_MESSAGE)
    try:
        await asyncio.to_thread(
            send_contact_email,
            contact_type=request.contact_type,
            name=request.name,
            email=request.email,
            message=request.message,
        )
    except Exception as e:
        print(f"Contact endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to send message. Please try again later."
        ) from e


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    RAG chat endpoint using Groq + ChromaDB.

    This is the core RAG (Retrieval-Augmented Generation) endpoint that:
    1. Takes a user query and conversation history
    2. Performs semantic search in ChromaDB to find relevant handbook chunks
    3. Sends the retrieved context to Groq openai/gpt-oss-20b
    4. Returns the generated answer with source citations

    The RAG service ensures all responses are grounded in actual handbook content.
    Uses Groq for primary generation with OpenAI as fallback.

    Args:
        request (ChatRequest): Contains query string and message history

    Returns:
        ChatResponse: Contains answer content and array of source citations

    Raises:
        HTTPException 503: If vector database is not initialized
        HTTPException 500: If RAG processing fails
    """
    try:
        # Verify RAG service is initialized (requires vector DB)
        if rag_service is None:
            raise HTTPException(
                status_code=503,
                detail="RAG service not initialized. Please ensure vector database is created.",
            )

        # chat_id groups all messages in one conversation session.
        # interaction_id uniquely identifies this single user/LLM exchange.
        chat_id = request.chat_id or str(uuid4())
        interaction_id = str(uuid4())
        start_time = time.time()

        # Daily caps: answered like a normal message so the UI shows it (and doesn't retry).
        if not await asyncio.to_thread(ratelimit.chat_allowed, http_request):
            return ChatResponse(
                content=ratelimit.CHAT_LIMIT_MESSAGE,
                sources=[],
                chat_id=chat_id,
                interaction_id=interaction_id,
            )

        # Perform RAG query: retrieve context + generate response
        result = await rag_service.get_rag_response(query=request.query, history=request.history)

        response_time_ms = int((time.time() - start_time) * 1000)

        await chat_logger.log_message_async(
            interaction_id=interaction_id,
            chat_id=chat_id,
            user_message=request.query,
            llm_response=result["content"],
            response_time_ms=response_time_ms,
        )

        raw_steps = result.get("tool_steps", [])
        tool_steps = [ToolStep(**s) for s in raw_steps] if raw_steps else None

        return ChatResponse(
            content=result["content"],
            sources=result["sources"],
            chat_id=chat_id,
            interaction_id=interaction_id,
            tool_steps=tool_steps,
        )
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """
    Streaming chat endpoint using Server-Sent Events.

    Sends tool_step events as the agent processes, then a final done event
    with the full answer, sources, and tool steps. While nothing else is sent, a
    ": ping" comment line goes out every HEARTBEAT_SECONDS (clients ignore it).
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized.",
        )

    chat_id = request.chat_id or str(uuid4())
    interaction_id = str(uuid4())
    start_time = time.time()

    if not await asyncio.to_thread(ratelimit.chat_allowed, http_request):
        # Daily caps: a normal (non-error) answer, so the UI shows it and doesn't retry.
        payload = {
            "content": ratelimit.CHAT_LIMIT_MESSAGE,
            "sources": [],
            "tool_steps": [],
            "chat_id": chat_id,
            "interaction_id": interaction_id,
        }

        async def limited():
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            limited(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def event_generator():
        final_content = ""
        try:
            async for event in _with_heartbeat(
                rag_service.get_rag_response_streamed(
                    query=request.query,
                    history=request.history,
                ),
                HEARTBEAT_SECONDS,
            ):
                if event is None:
                    yield ": ping\n\n"
                elif event["event"] == "tool_step":
                    yield f"event: tool_step\ndata: {json.dumps(event['data'])}\n\n"
                elif event["event"] == "done":
                    data = event["data"]
                    final_content = data.get("content", "")
                    payload = {
                        "content": data["content"],
                        "sources": data["sources"],
                        "tool_steps": data["tool_steps"],
                        "chat_id": chat_id,
                        "interaction_id": interaction_id,
                    }
                    if data.get("isError"):
                        payload["isError"] = True
                    yield f"event: done\ndata: {json.dumps(payload)}\n\n"
        except Exception as e:
            print(f"Stream error: {e}")
            payload = {
                "content": "I'm having trouble right now. Please try again.",
                "sources": [],
                "tool_steps": [],
                "chat_id": chat_id,
                "interaction_id": interaction_id,
                "isError": True,
            }
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"
        finally:
            response_time_ms = int((time.time() - start_time) * 1000)
            await chat_logger.log_message_async(
                interaction_id=interaction_id,
                chat_id=chat_id,
                user_message=request.query,
                llm_response=final_content,
                response_time_ms=response_time_ms,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# Frontend Static File Serving (Production Only)
# =============================================================================
#
# In production (Docker), the built React app is served from the same server.
# This section mounts the frontend dist folder and handles client-side routing.
#
# For local development:
# - Frontend runs on port 3000 (Vite dev server)
# - Backend runs on port 9481 (this FastAPI server)
# - CORS middleware allows cross-origin requests
#
# In production:
# - Backend serves API on /api/* routes
# - Backend serves frontend on all other routes
# - No CORS needed (same origin)

frontend_path = os.getenv("FRONTEND_PATH", "/app/frontend/dist")
if os.path.exists(frontend_path):
    # Mount static assets (JS, CSS, images) at /assets
    app.mount("/assets", StaticFiles(directory=f"{frontend_path}/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """
        Catch-all route for serving the React frontend.

        This enables client-side routing (React Router):
        - If a file exists at the requested path, serve it
        - Otherwise, serve index.html and let React handle routing

        Must be defined AFTER all API routes, otherwise it would catch /api/* requests.

        Args:
            full_path: Any path that didn't match an API route

        Returns:
            FileResponse: Either the requested file or index.html
        """
        # Try to serve specific file (e.g., favicon.ico, robots.txt)
        file_path = Path(frontend_path) / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Fall back to index.html for client-side routing
        # React Router will handle the actual route
        index_path = Path(frontend_path) / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)

        raise HTTPException(status_code=404, detail="Not found")
else:
    print(f"[WARNING] Frontend path not found: {frontend_path}. Running in API-only mode.")
