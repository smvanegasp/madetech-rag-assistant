"""The app's single DynamoDB table (pk/sk, TTL on "ttl"): chat logs and rate-limit counters.

Created by Terraform on AWS (infra/data.tf). Locally TABLE_NAME is unset and nothing is stored.
"""

import os
import time
from functools import cache

import boto3

from .settings import table_name


@cache
def get_table():
    resource = boto3.resource(
        "dynamodb",
        endpoint_url=os.environ.get("DYNAMODB_ENDPOINT") or None,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    return resource.Table(table_name())


def create_table_if_missing() -> None:
    """For tests and local DynamoDB only; on AWS the table comes from Terraform."""
    client = get_table().meta.client
    if table_name() in client.list_tables()["TableNames"]:
        return
    client.create_table(
        TableName=table_name(),
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def increment(key: str, ttl_seconds: int) -> int:
    """Atomically add 1 to a counter row and return the new value."""
    response = get_table().update_item(
        Key={"pk": f"COUNTER#{key}", "sk": "COUNTER"},
        UpdateExpression="ADD #n :one SET #ttl = if_not_exists(#ttl, :ttl)",
        ExpressionAttributeNames={"#n": "n", "#ttl": "ttl"},
        ExpressionAttributeValues={":one": 1, ":ttl": int(time.time()) + ttl_seconds},
        ReturnValues="UPDATED_NEW",
    )
    return int(response["Attributes"]["n"])
