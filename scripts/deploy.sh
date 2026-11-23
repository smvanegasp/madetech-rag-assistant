#!/usr/bin/env bash
# Deploy to AWS: API image -> ECR, Terraform apply, frontend -> S3, CloudFront invalidation.
# Merges into main deploy code automatically (.github/workflows/deploy.yml); run this by hand
# for infrastructure changes or a manual deploy.
# Run from Git Bash in the repo root:  scripts/deploy.sh
# Extra arguments go to `terraform apply` (e.g. -auto-approve).
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-nexus}" AWS_REGION=us-east-1
# Git Bash would otherwise rewrite arguments like "/*" into Windows paths.
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/.."

tf() { terraform -chdir=infra "$@"; }

echo "== Terraform init"
tf init -input=false >/dev/null

# The first deploy needs the image repository before there is anything to push to it.
if ! aws ecr describe-repositories --repository-names nexus-api >/dev/null 2>&1; then
  echo "== Creating the image repository"
  tf apply -input=false -target=aws_ecr_repository.api "$@"
fi

account=$(aws sts get-caller-identity --query Account --output text)
registry="$account.dkr.ecr.$AWS_REGION.amazonaws.com"
image="$registry/nexus-api:latest"

echo "== Building and pushing the API image"
aws ecr get-login-password | docker login --username AWS --password-stdin "$registry" >/dev/null
# Lambda wants a plain single-platform image (no provenance/SBOM attestations).
docker build --platform linux/amd64 --provenance=false --sbom=false --target lambda -t "$image" backend
docker push "$image"

echo "== Terraform apply"
tf apply -input=false "$@"

bucket=$(tf output -raw web_bucket)
distribution=$(tf output -raw distribution_id)

echo "== Building the frontend"
(cd frontend && npm ci --no-audit --no-fund && npm run build)

echo "== Uploading the frontend"
scripts/upload_frontend.sh "$bucket" "$distribution"

echo "== Done: $(tf output -raw url)"
