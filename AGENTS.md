# AGENTS.md: Nexus, the Made Tech handbook assistant

Read this first. It is the full context for continuing work in a new chat: what the app is, how
to run and test it, how it is deployed, and the rules that matter. **[README.md](README.md)** is
the public explanation; update it in the same PR whenever the architecture, CI/CD or AWS story
changes. Open ideas live in [IMPROVEMENTS.md](IMPROVEMENTS.md).

This app follows the same deployment template as Sergio's other apps (pantry-mgmt-app,
fit-mgmt-app, carhaul-ai-agent in `C:\Users\smvan\repos\`). **When in doubt, do it the way they
do**: Terraform in `infra/`, Lambda + Web Adapter, CloudFront, SSM secrets, DynamoDB, GitHub
Actions with OIDC. Copy and adapt their files rather than reinventing.

---

## 1. What this is

An agentic RAG chatbot over the 161-document Made Tech handbook, linked from sergiovanegas.com.

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + TypeScript + Vite (Tailwind and fonts from CDNs in `index.html`) |
| Backend | FastAPI on Lambda (container image + AWS Lambda Web Adapter), Python 3.12, uv |
| Agent | OpenAI Agents SDK, agent "Nexus", on Groq `openai/gpt-oss-20b` |
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector DB | Chroma Cloud (database `madetech_handbook`, collection `docs`) + in-memory BM25 |
| Chat logs, rate limits | DynamoDB table `nexus` |
| Email | Resend (contact/feedback form and agent tools) |

**Do not change the models, embeddings or Chroma collection casually**: the embedding model in
`backend/config.yaml` must match what was used at ingestion, and the visitor experience is meant
to stay stable. S3 Vectors is the AWS-native alternative to Chroma, to evaluate as its own change.

## 2. Local development

```bash
# Backend (from backend/)
uv sync                                    # creates .venv with dev tools
uv run uvicorn src.app:app --reload --port 9481

# Frontend (from frontend/), separate terminal
npm install
npm run dev                                # http://localhost:3000, proxies /api to :9481
```

Secrets go in `backend/.env` (see `backend/.env.example`). Without `TABLE_NAME`, chats aren't
logged and there are no rate limits, which is what you want locally.

Container (same image as AWS, minus the Lambda adapter):

```bash
docker build --target app -t nexus-api backend
docker run -p 9481:8080 --env-file backend/.env nexus-api
```

## 3. Tests and checks (what CI runs)

```bash
cd backend && uv run ruff check . && uv run ruff format --check . && uv run pytest -q
cd frontend && npm test && npm run build      # build includes tsc --noEmit
terraform fmt -check -recursive infra && terraform -chdir=infra validate
```

Backend tests use a fake RAG service and moto's in-memory DynamoDB: no model, Chroma or AWS
calls, no spend. `experiments/scripts/05_surprise_me_answers/` runs real questions against a
deployed URL, which is the way to compare answers before and after a change.

## 4. How a chat flows

`POST /api/chat/stream` (Server-Sent Events) -> rate-limit check (DynamoDB counters) ->
`RAGService.get_rag_response_streamed` -> the agent calls `search_handbook` / `plan_searches`
(semantic + BM25) -> one `event: tool_step` per tool call as it happens -> `event: done` with the
answer, sources and tool steps -> chat logged to DynamoDB. While the agent works, `: ping`
comment lines go out every 15s (CloudFront drops an origin that is silent for 60s). The frontend
parser (`frontend/services/sse.ts`) ignores them.

Hitting a daily cap returns a normal `done` answer (not `isError`), so the UI shows the message
and doesn't retry. Caps: 50 chats per IP per day, 1000 overall, 10 contact messages per IP.

## 5. AWS deployment

Live at **https://nexus.sergiovanegas.com** (the CloudFront URL works too). Account 816940506502,
us-east-1, AWS CLI profile `nexus`; a fully separate stack from the other apps: everything is
named `nexus-*`. Terraform state is in `nexus-tfstate-816940506502` (created by
`infra/bootstrap/`, whose own small state stays local and git-ignored). The alert email is in the
git-ignored `infra/terraform.tfvars` (`alert_email = "..."`).

- **What runs:** CloudFront (`/*` from the private bucket `nexus-web-<account>`, `/api/*` to the
  Lambda Function URL with the `X-Origin-Verify` secret) -> Lambda `nexus-api` (the
  `backend/Dockerfile` lambda stage, image in ECR `nexus-api`, last 5 kept; 2048 MB, 120s,
  reserved concurrency 10) -> Chroma Cloud, OpenAI, Groq, Resend, and DynamoDB `nexus`.
- **Why a Function URL and not API Gateway (unlike the other apps):** the URL runs in
  `RESPONSE_STREAM` mode with `AWS_LWA_INVOKE_MODE=response_stream`, so SSE events reach the
  browser live. API Gateway HTTP APIs buffer responses. The URL is public but the app returns 403
  without CloudFront's origin secret.
- **Secrets:** SSM SecureStrings under `/nexus/` (`GROQ_API_KEY`, `OPENAI_API_KEY`,
  `CHROMA_API_KEY`, `TENANT_CHROMA`, `RESEND_API_KEY`, `CONTACT_EMAIL`), loaded at cold start by
  `backend/src/ssm.py`. Set them with `cd backend && uv run python ../scripts/set_aws_secrets.py`
  (reads `.env`, asks for anything missing, restarts the Lambda). Never print them.
- **Alarms** (SNS `nexus-alerts` -> email, after the subscription is confirmed): chats failing
  ("Stream error" log line, 3+ in 15 min), contact email failing,
  Lambda errors, throttles, and slow requests (p90 > 60s). Keep those log lines stable.
- **Chat logs:** DynamoDB items `pk=CHAT#<chat_id>`, `sk=MSG#<iso time>#<interaction_id>` with
  `user_message`, `llm_response`, `response_time_ms`; deleted by TTL after 180 days.
- **Everything deploys on merge (unlike the other apps, infrastructure too):** after CI passes
  on `main`, `.github/workflows/deploy.yml` assumes `nexus-github-deploy` through GitHub OIDC (no
  AWS keys in GitHub; only workflows on main, matched by the repo's immutable subject; the role
  has AdministratorAccess because it applies Terraform), pushes the image, runs
  `terraform apply` (which also rolls Lambda to the new image digest), uploads the frontend and
  checks `/api/health`. **Merging a PR is the approval** for its infrastructure changes.
- **PRs show the plan:** the CI job `infra-plan` assumes the read-only `nexus-github-plan` role
  and writes `terraform plan` (no lock) to the job summary. Read it before merging any PR that
  touches `infra/`. Skipped for Dependabot and forks (GitHub gives them no OIDC token).
- **Repo variables:** `DEPLOY_ENABLED=true` (turns on deploys and PR plans) and `ALERT_EMAIL`
  (Terraform's `alert_email`; marked sensitive so it stays out of the public logs).
- **Deploy by hand** (first deploy, or if Actions is down): `scripts/deploy.sh` from Git Bash
  (image -> ECR, `terraform apply`, frontend -> S3, invalidation), using profile `nexus`. For
  local Terraform commands set `AWS_PROFILE=nexus` (the config has no profile so CI can run it).
  The agent may run `terraform plan`; **local applies need Sergio's OK.**
- **Domain** (`infra/domain.tf`): DNS is at Hostinger, so Terraform never touches it. Two CNAMEs
  there: ACM's validation record (output `certificate_validation_record`; keep it, the
  certificate renews through it) and `nexus` -> the CloudFront domain (output `dns_target`).

### First-time setup (once)

1. AWS CLI profile `nexus` for account 816940506502 (an IAM user like `pantry-deployer`).
2. Turn on GitHub's immutable OIDC subject for this repo (the deploy role trusts that form):
   `gh api -X PUT repos/smvanegasp/madetech-rag-assistant/actions/oidc/customization/sub -F use_default=true -F use_immutable_subject=true`
3. `terraform -chdir=infra/bootstrap init && terraform -chdir=infra/bootstrap apply`
4. `infra/terraform.tfvars` with `alert_email`.
5. `cd backend && uv run python ../scripts/set_aws_secrets.py` (before the Lambda exists it
   only stores the secrets, so the first start already finds them).
6. `scripts/deploy.sh` (Git Bash). The first apply waits (up to 2h) for the certificate: add the
   validation CNAME at Hostinger as soon as it is printed (see below), then the `nexus` CNAME.
   This local run is what creates the GitHub roles; every later deploy happens on merge.
7. Repo variables `DEPLOY_ENABLED=true` and `ALERT_EMAIL`.

## 6. Handbook data and ingestion

Source markdown: `backend/data/handbook/` (161 files by category). They are baked into the image
(the `/api/handbook` source viewer and BM25 use them). Re-ingesting into Chroma:
`cd backend && uv sync --group ingest && uv run python scripts/01_llm_chunking_embedding/main.py`.
It deletes and recreates the `docs` collection, so search is briefly empty while it runs.

## 7. Rules

- Never print or commit secrets; the repo is public and CI runs gitleaks.
- `terraform plan` is fine; `terraform apply`, deleting AWS resources and changing repo settings
  need Sergio's OK. Sergio merges PRs.
- Keep `openai` on 2.x until an upgrade is tested on purpose (`backend/pyproject.toml`).
- Keep `utils/prompts.py` text changes deliberate: they change what visitors see.
