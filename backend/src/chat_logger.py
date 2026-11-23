"""Chat message logger: stores each user/LLM interaction in DynamoDB.

One item per interaction: pk = CHAT#<chat_id>, sk = MSG#<iso timestamp>#<interaction_id>,
expiring after CHAT_LOG_DAYS through the table's TTL. Read a conversation back with a Query on
its pk. Off when TABLE_NAME is unset (local dev). Errors are logged, never raised, so logging
can't break a chat.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime

from . import db
from .settings import CHAT_LOG_DAYS, table_name

logger = logging.getLogger(__name__)


def log_message(
    interaction_id: str,
    chat_id: str,
    user_message: str,
    llm_response: str,
    response_time_ms: int,
) -> None:
    """Store one interaction. Never raises."""
    if not table_name():
        return
    timestamp = datetime.now(UTC).isoformat()
    try:
        db.get_table().put_item(
            Item={
                "pk": f"CHAT#{chat_id}",
                "sk": f"MSG#{timestamp}#{interaction_id}",
                "interaction_id": interaction_id,
                "chat_id": chat_id,
                "timestamp": timestamp,
                "user_message": user_message,
                "llm_response": llm_response,
                "response_time_ms": response_time_ms,
                "ttl": int(time.time()) + CHAT_LOG_DAYS * 86400,
            }
        )
    except Exception as e:
        logger.error("Failed to log chat message: %s", e)


async def log_message_async(**kwargs) -> None:
    """log_message off the event loop, so a slow write never stalls other requests."""
    await asyncio.to_thread(log_message, **kwargs)
