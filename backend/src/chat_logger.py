"""Chat message logger — persists each user/LLM interaction to Supabase (PostgreSQL).

Reads DATABASE_URL from the environment. All errors are caught and logged
to stdout so that a DB failure never breaks the /api/chat endpoint.
"""

import logging
import os
from datetime import datetime, timezone

import psycopg2

logger = logging.getLogger(__name__)

_conn = None


def _get_connection():
    """Return a cached psycopg2 connection, reconnecting if closed."""
    global _conn
    if _conn is None or _conn.closed:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        _conn = psycopg2.connect(database_url)
        _conn.autocommit = True
    return _conn


def log_message(
    interaction_id: str,
    chat_id: str,
    user_message: str,
    llm_response: str,
    response_time_ms: int,
) -> None:
    """Insert one interaction row into chat_messages.

    Never raises — errors are logged so the chat endpoint is never broken
    by a database failure.
    """
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages
                    (interaction_id, chat_id, timestamp, user_message, llm_response, response_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    interaction_id,
                    chat_id,
                    datetime.now(timezone.utc),
                    user_message,
                    llm_response,
                    response_time_ms,
                ),
            )
    except Exception as e:
        logger.error("Failed to log chat message: %s", e)
