# Stage "app": the API image, for running locally and in CI.
#   docker build --target app -t nexus-api backend
#   docker run -p 9481:8080 --env-file .env nexus-api
FROM python:3.12-slim AS app

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Code, the RAG config and the handbook (loaded into memory at startup for /api/handbook and
# keyword search). The vectors live in Chroma Cloud.
COPY config.yaml ./
COPY src ./src
COPY utils ./utils
COPY data/handbook ./data/handbook

EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]

# Stage "lambda" (default): the same image plus the Lambda Web Adapter extension, which
# forwards Lambda invocations to uvicorn so FastAPI runs unchanged on AWS. Responses stream
# (Function URL in RESPONSE_STREAM mode + AWS_LWA_INVOKE_MODE), so SSE events arrive live.
FROM app AS lambda
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
