"""Secrets from SSM Parameter Store on AWS.

Lambda gets SSM_PREFIX ("/nexus/"); every parameter under it becomes an environment variable of
the same name at startup ("/nexus/GROQ_API_KEY" -> GROQ_API_KEY), so the rest of the app reads
settings the same way locally (.env) and on AWS. Variables already set win.
"""

import os

import boto3


def load_parameters() -> None:
    prefix = os.environ.get("SSM_PREFIX")
    if not prefix:
        return
    ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    pages = ssm.get_paginator("get_parameters_by_path").paginate(Path=prefix, WithDecryption=True)
    for page in pages:
        for param in page["Parameters"]:
            os.environ.setdefault(param["Name"].removeprefix(prefix), param["Value"])
