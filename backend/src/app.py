"""
@file app.py
@description FastAPI application serving the RAG handbook backend.

This is the main application entry point that orchestrates the entire backend system.
It provides REST API endpoints for chat, document highlighting, and static file serving.

Architecture:
- Chat responses: Uses RAGService (Groq + ChromaDB) for semantic search
- Highlights: Uses Groq service for precise text extraction
- Frontend: Serves built React app in production (Docker)

Required environment variables:
- GROQ_API_KEY: For primary LLM (chat and highlighting)
- OPENAI_API_KEY: For embeddings and fallback LLM
- GEMINI_API_KEY: For fallback highlighting (optional)
- FRONTEND_PATH: Path to built frontend (optional, defaults to /app/frontend/dist)
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from .models import (
    ChatRequest, ChatResponse, 
    HighlightsRequest, HighlightsResponse,
    HandbookDoc
)
from .groq_service import get_relevance_highlights
from .handbook_loader import load_handbook_documents
from .rag_service import RAGService

# Load environment variables from .env or .env.local
load_dotenv()

# Initialize FastAPI app with OpenAPI documentation
app = FastAPI(
    title="RAG Company Handbook API",
    description="Backend API for the RAG Company Handbook chatbot",
    version="1.0.0"
)

# CORS middleware for local development
# Allows frontend dev server (Vite) to connect to backend
# In production, frontend is served from same origin, so CORS not needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state: Loaded at startup and cached in memory
# These are read-only after initialization, so thread-safe
handbook_docs: list[HandbookDoc] = []
rag_service: RAGService = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize application state on server startup.
    
    This function runs once when the FastAPI server starts. It:
    1. Loads all markdown handbook documents from data/handbook/
    2. Initializes the RAG service with ChromaDB connection
    3. Validates that the vector database exists
    
    If the vector database is missing, the server will fail to start.
    To create it, run: python -m scripts.ingest
    """
    global handbook_docs, rag_service
    try:
        # Load all handbook documents (used by /api/handbook endpoint)
        handbook_docs = load_handbook_documents()
        print(f"✓ Loaded {len(handbook_docs)} handbook documents")
        
        # Initialize RAG service with vector database
        # Vector DB must exist at backend/data/vector_db/ (created by scripts/ingest.py)
        import os
        vector_db_path = os.path.join(os.path.dirname(__file__), "..", "data", "vector_db")
        rag_service = RAGService(vector_db_path=vector_db_path)
        print(f"✓ RAG service initialized with Groq LLM and ChromaDB at {vector_db_path}")
    except Exception as e:
        print(f"✗ Error during startup: {e}")
        raise


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns server status and the count of loaded handbook documents.
    Useful for monitoring, container orchestration, and debugging.
    
    Returns:
        dict: {"status": "healthy", "documents_loaded": <count>}
    """
    return {
        "status": "healthy",
        "documents_loaded": len(handbook_docs)
    }


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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
                detail="RAG service not initialized. Please ensure vector database is created."
            )
        
        # Perform RAG query: retrieve context + generate response
        result = await rag_service.get_rag_response(
            query=request.query,
            history=request.history
        )
        
        return ChatResponse(
            content=result["content"],
            sources=result["sources"]
        )
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/highlights", response_model=HighlightsResponse)
async def get_highlights(request: HighlightsRequest):
    """
    Semantic analysis endpoint using Groq.
    
    This endpoint provides the "Highlight with AI" feature. When a user views
    a source document, they can request AI-powered highlighting to see which
    specific phrases support the chatbot's previous answer.
    
    Uses Groq for fast inference with Gemini as fallback for reliability.
    
    Process:
    1. Receives the AI's previous answer and full document content
    2. Asks Groq to find 5-8 exact phrases in the document
    3. Returns array of verbatim strings that support the claims
    4. Frontend injects <mark> tags around these phrases
    
    Args:
        request (HighlightsRequest): Contains answer and document_content
        
    Returns:
        HighlightsResponse: Contains array of verbatim text snippets to highlight
        
    Raises:
        HTTPException 500: If both Groq and Gemini fail
    """
    try:
        highlights = await get_relevance_highlights(
            answer=request.answer,
            document_content=request.document_content
        )
        return HighlightsResponse(highlights=highlights)
    except Exception as e:
        print(f"Highlights endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    print(f"⚠ Frontend path not found: {frontend_path}. Running in API-only mode.")
