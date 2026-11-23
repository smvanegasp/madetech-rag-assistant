"""Daily caps on chats and contact messages, kept in DynamoDB so they hold across Lambda instances.

Off when TABLE_NAME is unset (local dev). A failing counter never blocks a visitor.
"""

import logging
import time

from fastapi import Request

from . import db
from .settings import (
    chats_per_day,
    chats_per_ip_per_day,
    contacts_per_ip_per_day,
    table_name,
)

logger = logging.getLogger(__name__)

DAY = 86400

CHAT_LIMIT_MESSAGE = (
    "You've reached today's limit of questions for this demo. "
    "Please come back tomorrow - thanks for trying Nexus!"
)
CONTACT_LIMIT_MESSAGE = "Too many messages sent today. Please try again tomorrow."


def client_ip(request: Request) -> str:
    # Behind CloudFront the viewer's address arrives in this header ("ip:port").
    viewer = request.headers.get("cloudfront-viewer-address")
    if viewer:
        return viewer.rsplit(":", 1)[0]
    return request.client.host if request.client else "unknown"


def _hit(name: str) -> int:
    window = int(time.time()) // DAY
    return db.increment(f"{name}#{window}", ttl_seconds=DAY * 2)


def chat_allowed(request: Request) -> bool:
    """Count one chat for this visitor and overall; False once either daily cap is passed."""
    if not table_name():
        return True
    try:
        if _hit(f"chat-ip#{client_ip(request)}") > chats_per_ip_per_day():
            logger.warning("Chat rate limit reached (per IP)")
            return False
        if _hit("chat-all") > chats_per_day():
            logger.warning("Chat rate limit reached (global)")
            return False
    except Exception as e:  # never lock visitors out because a counter failed
        logger.error("Rate limit check failed: %s", e)
    return True


def contact_allowed(request: Request) -> bool:
    if not table_name():
        return True
    try:
        if _hit(f"contact-ip#{client_ip(request)}") > contacts_per_ip_per_day():
            logger.warning("Contact rate limit reached (per IP)")
            return False
    except Exception as e:
        logger.error("Rate limit check failed: %s", e)
    return True
