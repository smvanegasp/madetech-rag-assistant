"""Put the app's secrets in SSM Parameter Store for the AWS deployment.

Run from the repo root:  cd backend && uv run python ../scripts/set_aws_secrets.py

Each key is copied from backend/.env (or the repo-root .env) when present there, otherwise
asked for with a hidden prompt. Nothing secret is printed. Uses the "nexus" AWS profile unless
AWS_PROFILE is set. If the API is already deployed, it is restarted so it reads the new values.
"""

import getpass
import os
import sys
from pathlib import Path

import boto3
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "/nexus/"
FUNCTION = "nexus-api"
KEYS = [
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "CHROMA_API_KEY",
    "TENANT_CHROMA",
    "RESEND_API_KEY",
    "CONTACT_EMAIL",
]


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in (ROOT / ".env", ROOT / "backend" / ".env"):  # backend/.env wins
        if path.exists():
            values.update({k: v for k, v in dotenv_values(path).items() if v})
    return values


def main() -> None:
    session = boto3.Session(
        profile_name=os.environ.get("AWS_PROFILE", "nexus"), region_name="us-east-1"
    )
    ssm = session.client("ssm")
    env = read_env()

    for name in KEYS:
        value = env.get(name) or getpass.getpass(f"{name} (hidden): ")
        if not value:
            sys.exit(f"{name} is empty; nothing saved for it.")
        ssm.put_parameter(Name=PREFIX + name, Value=value, Type="SecureString", Overwrite=True)
        print(f"Saved {PREFIX}{name}")

    # Lambda reads SSM at cold start: redeploying the same image replaces warm instances.
    lam = session.client("lambda")
    try:
        image = lam.get_function(FunctionName=FUNCTION)["Code"]["ImageUri"]
    except lam.exceptions.ResourceNotFoundException:
        return
    lam.update_function_code(FunctionName=FUNCTION, ImageUri=image)
    print("Restarted the API so it picks up the new values.")


if __name__ == "__main__":
    main()
