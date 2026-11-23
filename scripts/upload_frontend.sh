#!/usr/bin/env bash
# Upload frontend/dist to the web bucket and invalidate CloudFront.
# Used by scripts/deploy.sh and .github/workflows/deploy.yml.
#   scripts/upload_frontend.sh <bucket> <distribution-id>
set -euo pipefail

bucket=$1
distribution=$2
cd "$(dirname "$0")/.."

# Hashed bundles never change; older ones stay so open tabs can still load their chunks.
aws s3 sync frontend/dist/assets "s3://$bucket/assets" \
  --cache-control "public,max-age=31536000,immutable" --only-show-errors
aws s3 sync frontend/dist "s3://$bucket" --delete --exclude "assets/*" --exclude index.html \
  --cache-control "public,max-age=3600" --only-show-errors
aws s3 cp frontend/dist/index.html "s3://$bucket/index.html" \
  --cache-control "no-cache" --only-show-errors

aws cloudfront create-invalidation --distribution-id "$distribution" --paths "/*" \
  --query Invalidation.Id --output text >/dev/null
