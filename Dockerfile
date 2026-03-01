# =======================================================================================
# Multi-stage Docker build for RAG Company Handbook
# =======================================================================================
#
# This Dockerfile creates a production-ready image with:
# - Built React frontend (stage 1)
# - Python backend with all dependencies (stage 2)
# - Minimal runtime image with non-root user (stage 3)
#
# Build arguments:
# - SPACE_HOST: Optional Hugging Face Space hostname for production deployment
#
# Build command:
#   docker build -t rag-handbook .
#   docker build --build-arg SPACE_HOST=your-space.hf.space -t rag-handbook .
#
# Run command:
#   docker run -p 9481:9481 --env-file .env.local rag-handbook
#
# =======================================================================================


# =======================================================================================
# Stage 1: Frontend Build
# =======================================================================================
# Purpose: Build the React TypeScript frontend using Vite
# Output: Static files in /app/frontend/dist
FROM node:22-slim AS frontend-builder

# Build argument for Hugging Face Spaces deployment
# If provided, sets VITE_BACKEND_URL to the Space's HTTPS URL
# If not provided, uses empty string (same-origin) for production
ARG SPACE_HOST

# Enable corepack and install pnpm package manager
# pnpm is faster and more disk-efficient than npm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Set working directory for frontend build
WORKDIR /app/frontend

# Copy package.json first for Docker layer caching
# If package.json hasn't changed, Docker reuses cached dependencies
COPY frontend/package.json ./

# Install dependencies
# Try frozen-lockfile first (fails if lock is out of sync)
# Fall back to regular install if frozen fails
RUN pnpm install --frozen-lockfile || pnpm install

# Copy all frontend source files
COPY frontend/ ./

# Build the frontend with Vite
# Conditional VITE_BACKEND_URL based on SPACE_HOST:
# - With SPACE_HOST: Uses https://your-space.hf.space
# - Without SPACE_HOST: Uses empty string (relative URLs for same-origin)
ENV VITE_APP_ENV=production
RUN if [ -n "$SPACE_HOST" ]; then \
        echo "Building with SPACE_HOST: $SPACE_HOST"; \
        VITE_BACKEND_URL="https://$SPACE_HOST" pnpm build; \
    else \
        echo "Building without SPACE_HOST (using same-origin URLs)"; \
        VITE_BACKEND_URL="" pnpm build; \
    fi

# Build output: /app/frontend/dist/ contains static HTML, JS, CSS




# =======================================================================================
# Stage 2: Backend Build
# =======================================================================================
# Purpose: Install Python dependencies and prepare backend + data
# Output: Backend code, dependencies, and built frontend in /app
FROM python:3.12-slim AS backend-builder

# Install uv - a fast Python package installer (alternative to pip)
# uv is 10-100x faster than pip for dependency resolution
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy backend source code and data
# This includes:
# - src/ (Python application code)
# - data/handbook/ (markdown documents)
# - data/vector_db/ (pre-created ChromaDB database)
# - pyproject.toml (dependency specification)
COPY backend/ ./backend/

# Verify critical data directories exist before proceeding
# The application cannot function without these directories
RUN test -d backend/data/handbook || (echo "ERROR: handbook directory not found" && exit 1)
RUN test -d backend/data/vector_db || (echo "ERROR: vector_db directory not found" && exit 1)

# Install Python dependencies using uv
# --no-cache-dir: Don't store cache (reduces image size)
# --system: Install to system Python (no virtual environment needed in container)
RUN cd backend && uv pip install --no-cache-dir --system .

# Copy built frontend from stage 1
# The backend will serve these static files in production
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist



# =======================================================================================
# Stage 3: Production Runtime
# =======================================================================================
# Purpose: Create minimal production image with security best practices
# Output: Final image with backend, frontend, and data (~500MB)
FROM python:3.12-slim

# Security: Create non-root user to run the application
# Running as root in containers is a security risk
RUN useradd -m appuser

# Create temporary data directory for ChromaDB operations
# Some ChromaDB operations require a writable temp directory
RUN mkdir -p /app/tmp_data && chown appuser:appuser /app/tmp_data

WORKDIR /app

# Copy Python packages from builder stage
# This includes all installed dependencies from pyproject.toml
COPY --from=backend-builder /usr/local /usr/local

# Copy application code, data, and built frontend from builder stage
# /app/backend/src/ - Python application
# /app/backend/data/ - Handbook markdown and vector database
# /app/frontend/dist/ - Built React app
COPY --from=backend-builder /app /app

# Verification: Ensure all required data is present in final image
# This catches issues early if data directories were accidentally excluded
# Also displays a count of handbook files for debugging
RUN python -c "import os; \
    handbook_path = '/app/backend/data/handbook'; \
    vector_db_path = '/app/backend/data/vector_db'; \
    assert os.path.exists(handbook_path), f'Handbook not found at {handbook_path}'; \
    assert os.path.exists(vector_db_path), f'Vector DB not found at {vector_db_path}'; \
    import glob; \
    md_count = len(glob.glob(handbook_path + '/**/*.md', recursive=True)); \
    print(f'✓ Data directories verified: {md_count} handbook files found')"

# ChromaDB requires write access to its database directory
# SQLite WAL mode writes to the database even for read queries
# Also grant write access to tmp_data for temporary operations
RUN chown -R appuser:appuser /app/backend/data/vector_db /app/tmp_data

# Environment variable: Tell FastAPI where to find built frontend
# The backend serves these files for production (no separate frontend server)
ENV FRONTEND_PATH=/app/frontend/dist

# Suppress Hugging Face hub warnings (optional, reduces noise in logs)
ENV HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

# Switch to non-root user for security
# All subsequent commands and the application run as appuser
USER appuser

# Expose port 9481 for the FastAPI application
# Note: This is declarative (doesn't actually publish the port)
# Use -p 9481:9481 when running the container
EXPOSE 9481

# Run the FastAPI application using uvicorn ASGI server
# --host 0.0.0.0: Listen on all network interfaces (required for Docker)
# --port 9481: Listen on port 9481
# backend.src.app:app: Module path and FastAPI app instance
CMD ["uvicorn", "backend.src.app:app", "--host", "0.0.0.0", "--port", "9481"]
