"""
@file rag_service.py
@description RAG (Retrieval-Augmented Generation) service using Groq + ChromaDB.

This service implements the core RAG pipeline for the handbook chatbot:
1. Semantic search: Find relevant document chunks in ChromaDB
2. Context assembly: Combine retrieved chunks into a prompt
3. LLM generation: Use Groq to generate an answer with the context
4. Source extraction: Map retrieved chunks to citation objects

Architecture:
- Embedding model: OpenAI text-embedding-3-large (3072 dimensions)
- LLM: Groq openai/gpt-oss-20b (temperature=0 for consistency)
- Fallback LLM: OpenAI gpt-4o-mini (if Groq fails)
- Vector DB: ChromaDB with cosine similarity
- Retrieval: Top-k chunks (k=10) with conversation history

Required environment variables:
- GROQ_API_KEY: For primary LLM generation
- OPENAI_API_KEY: For embeddings and fallback LLM
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

from .models import Message, SourceChunk

load_dotenv(override=True)

# Configuration constants
GROQ_MODEL = "openai/gpt-oss-20b"  # Groq model for chat completion
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI fallback model
RETRIEVAL_K = 10  # Number of document chunks to retrieve per query

SYSTEM_PROMPT = """You are "Nexus AI", an elite corporate knowledge assistant.
Your primary function is to answer employee questions using ONLY the provided handbook documents.

IMPORTANT: All source documents will be displayed to users below your response. Users can click on them to verify your answer against the original documentation.

STRICT CITATION RULES:
1. **Source Fidelity**: Use ONLY the information provided in the context below.
2. **Comprehensive Coverage**: Most topics are split across multiple sections. Reference ALL relevant information from the context.
3. **Accuracy First**: When making claims, directly reference or quote exact phrases from the context to ensure accuracy.
4. **Markdown Formatting**: Use professional Markdown for headers, lists, and tables. Ensure readability.
5. **Clarity**: Structure your answer logically with clear sections when covering multiple aspects of a topic.

If the context doesn't contain enough information to answer the question, say so clearly and explain what information is missing.

CONTEXT:
{context}
"""


class RAGService:
    """
    RAG service for handbook queries using Chroma vector database and Groq.
    
    This class encapsulates the entire RAG pipeline. It maintains a persistent
    connection to the ChromaDB vector store and uses:
    - OpenAI embeddings: Convert queries to vectors for semantic search
    - Groq LLM: Generate answers using retrieved context
    - OpenAI LLM: Fallback if Groq fails
    
    The service is initialized once at application startup and reused for all queries.
    """
    
    def __init__(self, vector_db_path: str):
        """
        Initialize the RAG service with a Chroma vector database.
        
        This constructor:
        1. Validates that the vector database exists
        2. Connects to ChromaDB at the specified path
        3. Initializes Groq LLM client (primary)
        4. Initializes OpenAI client (embeddings + fallback LLM)
        5. Creates a retriever with k=10 configuration
        
        The vector database must be pre-created using scripts/ingest.py
        before starting the application.
        
        Args:
            vector_db_path: Path to the Chroma vector database directory
            
        Raises:
            FileNotFoundError: If vector database doesn't exist at path
            ValueError: If required API keys are not set
        """
        self.vector_db_path = vector_db_path
        
        # Initialize Groq client for primary LLM (requires GROQ_API_KEY)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.groq_client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Initialize OpenAI client for embeddings and fallback LLM
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize OpenAI embeddings (CRITICAL: must not change for vector DB compatibility)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Validate that vector database exists before connecting
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(
                f"Vector database not found at {vector_db_path}. "
                "Please run the ingest script first: python -m scripts.ingest"
            )
        
        # Connect to existing ChromaDB database
        # This loads the pre-computed embeddings without re-embedding
        self.vectorstore = Chroma(
            persist_directory=vector_db_path, 
            embedding_function=self.embeddings
        )
        
        # Create retriever that returns top-k similar chunks
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        print(f"✓ RAG service initialized with Groq LLM and vector database at {vector_db_path}")
    
    def _fetch_context(self, question: str) -> List[Document]:
        """
        Retrieve relevant context documents for a question using semantic search.
        
        This method:
        1. Converts the question to an embedding vector
        2. Performs cosine similarity search in ChromaDB
        3. Returns top-k most similar document chunks
        
        Args:
            question: User's question (can be combined with history)
            
        Returns:
            List of Document objects with page_content and metadata
        """
        return self.retriever.invoke(question)
    
    def _combined_question(self, question: str, history: List[Message]) -> str:
        """
        Combine all the user's messages into a single string for better retrieval.
        
        Problem: If we only search with the latest question, we might miss context
        from earlier in the conversation (e.g., "What about vacation days?").
        
        Solution: Concatenate all previous user questions with the current one.
        This gives the retriever more context to find relevant chunks.
        
        Example:
        - User: "Tell me about benefits"
        - User: "What about vacation days?"
        - Combined: "Tell me about benefits\nWhat about vacation days?"
        
        Args:
            question: Current question
            history: Conversation history (includes both user and assistant messages)
            
        Returns:
            Combined question string (only user messages)
        """
        # Extract only user messages from history
        prior = "\n".join(m.content for m in history if m.role == "user")
        if prior:
            return prior + "\n" + question
        return question
    
    def _generate_with_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using Groq LLM with OpenAI fallback.
        
        Attempts to use Groq first for fast inference. If Groq fails,
        falls back to OpenAI for reliability.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Generated answer text
            
        Raises:
            Exception: If both Groq and OpenAI fail
        """
        try:
            # Try Groq first (primary LLM)
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0  # Consistent responses
            )
            answer = response.choices[0].message.content
            print("✓ Response generated with Groq")
            return answer
            
        except Exception as groq_error:
            print(f"⚠ Groq failed, falling back to OpenAI: {groq_error}")
            
            try:
                # Fallback to OpenAI
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0
                )
                answer = response.choices[0].message.content
                print("✓ Response generated with OpenAI (fallback)")
                return answer
                
            except Exception as openai_error:
                print(f"✗ OpenAI fallback failed: {openai_error}")
                raise Exception(f"Both LLMs failed. Groq: {groq_error}, OpenAI: {openai_error}")
    
    async def get_rag_response(
        self, 
        query: str, 
        history: List[Message]
    ) -> Dict[str, Any]:
        """
        Generate a RAG-based response to a user query.
        
        This is the main RAG pipeline that orchestrates:
        1. **Retrieval**: Search ChromaDB for relevant chunks
        2. **Augmentation**: Inject chunks into system prompt
        3. **Generation**: Use Groq (or OpenAI fallback) to create an answer
        4. **Citation**: Extract source references
        
        The system prompt instructs the LLM to only use information from
        the provided context, preventing hallucinations.
        
        Args:
            query: User's current question
            history: Conversation history (for context and continuity)
            
        Returns:
            Dictionary with:
            - 'content': The generated answer (markdown formatted)
            - 'sources': Array of SourceChunk objects with docId and snippet
        """
        try:
            # Step 1: Combine query with conversation history
            # This improves retrieval when questions refer to previous context
            combined = self._combined_question(query, history)
            
            # Step 2: Retrieve relevant document chunks from ChromaDB
            docs = self._fetch_context(combined)
            
            # Handle case where no relevant documents found
            if not docs:
                return {
                    "content": "I couldn't find any relevant information in the handbook to answer your question. Could you rephrase or ask something else?",
                    "sources": []
                }
            
            # Step 3: Build context string from retrieved chunks
            # Chunks are separated by double newlines for readability
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Step 4: Create system prompt with embedded context
            # This is the "Augmentation" part of RAG
            system_prompt = SYSTEM_PROMPT.format(context=context)
            
            # Step 5: Build message array for chat completion (OpenAI format)
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (last 5 messages) for continuity
            # This allows follow-up questions like "tell me more about that"
            for msg in history[-5:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # Step 6: Generate response using Groq with OpenAI fallback
            answer = self._generate_with_llm(messages)
            
            # Step 7: Extract source citations from retrieved documents
            # Maps Document objects to SourceChunk objects for frontend
            sources = self._extract_sources(docs, answer)
            
            return {
                "content": answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"RAG service error: {e}")
            # Fail gracefully with user-friendly error message
            return {
                "content": "I'm having trouble processing your request right now. Please try again in a moment.",
                "sources": []
            }
    
    def _extract_sources(self, docs: List[Document], answer: str) -> List[SourceChunk]:
        """
        Extract source citations from retrieved documents.
        
        This method converts ChromaDB Document objects into SourceChunk objects
        that the frontend can display as citations. Each source includes:
        - docId: Unique identifier to link to full document
        - snippet: The actual text chunk that was retrieved
        
        Design decisions:
        - Show full chunk content (not truncated) for context
        - Allow multiple chunks from same document
        - Limit to top 10 to avoid UI overload
        - Truncate only if extremely long (>800 chars)
        
        Args:
            docs: Retrieved document chunks from ChromaDB
            answer: Generated answer (currently unused, but available for future filtering)
            
        Returns:
            List of SourceChunk objects with docId and snippet
        """
        sources = []
        
        for doc in docs:
            # Extract document ID from metadata (set during ingestion)
            # Fallback to source_file if doc_id not present
            doc_id = doc.metadata.get("doc_id", doc.metadata.get("source_file", "unknown"))
            
            # Use full chunk content as snippet
            # This gives users more context than a truncated excerpt
            snippet = doc.page_content.strip()
            
            # Only truncate if extremely long (> 800 chars)
            # Try to truncate at sentence boundary for readability
            if len(snippet) > 800:
                truncate_at = snippet.rfind('.', 0, 800)
                if truncate_at > 400:  # Ensure we keep at least half
                    snippet = snippet[:truncate_at + 1]
                else:
                    # No good sentence boundary, hard truncate
                    snippet = snippet[:800] + "..."
            
            sources.append(SourceChunk(
                docId=doc_id,
                snippet=snippet
            ))
            
            # Limit to top 10 chunks to avoid overwhelming the UI
            # Note: Multiple chunks from the same document are allowed
            # This is intentional - if multiple sections are relevant, show all
            if len(sources) >= 10:
                break
        
        return sources
