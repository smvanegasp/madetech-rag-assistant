"""Runtime settings from environment variables (.env locally, Lambda env + SSM on AWS)."""

import os


def on_aws() -> bool:
    return bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


def origin_secret() -> str | None:
    """Set on AWS: CloudFront sends it in X-Origin-Verify, so direct Function URL calls fail."""
    return os.environ.get("ORIGIN_SECRET") or None


def table_name() -> str | None:
    """The DynamoDB table for chat logs and rate-limit counters; unset locally (both off)."""
    return os.environ.get("TABLE_NAME") or None


def chats_per_ip_per_day() -> int:
    return int(os.environ.get("CHATS_PER_IP_PER_DAY", "50"))


def chats_per_day() -> int:
    return int(os.environ.get("CHATS_PER_DAY", "1000"))


def contacts_per_ip_per_day() -> int:
    return int(os.environ.get("CONTACTS_PER_IP_PER_DAY", "10"))


# How long a conversation log is kept in DynamoDB before TTL deletes it.
CHAT_LOG_DAYS = 180
