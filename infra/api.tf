# The FastAPI backend: the backend/Dockerfile "lambda" stage on Lambda, behind a Function URL
# in RESPONSE_STREAM mode so /api/chat/stream's Server-Sent Events reach the browser as they
# happen. (An API Gateway HTTP API, as in the other apps, would buffer the whole response and
# the live tool steps would all arrive at the end.)

resource "aws_ecr_repository" "api" {
  name = "${local.name}-api"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 5 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

# scripts/deploy.sh pushes ":latest"; pinning the digest makes Lambda update when it changes.
data "aws_ecr_image" "api" {
  repository_name = aws_ecr_repository.api.name
  image_tag       = "latest"
}

# CloudFront sends this header to the API; the app refuses requests without it, so the public
# Function URL is useless to call directly.
resource "random_password" "origin_secret" {
  length  = 40
  special = false
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.name}-api"
  retention_in_days = 30
}

resource "aws_iam_role" "api" {
  name = "${local.name}-api"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "api" {
  role = aws_iam_role.api.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.api.arn}:*"
      },
      {
        # Chat logs (PutItem) and rate-limit counters (UpdateItem).
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.main.arn
      },
      {
        # SecureStrings use the AWS-managed aws/ssm key, whose policy already allows SSM use.
        Effect = "Allow"
        Action = "ssm:GetParametersByPath"
        # The path itself, trailing slash included, is what GetParametersByPath is checked against.
        Resource = "arn:aws:ssm:us-east-1:${local.account_id}:parameter${local.ssm_prefix}"
      },
    ]
  })
}

resource "aws_lambda_function" "api" {
  function_name = "${local.name}-api"
  role          = aws_iam_role.api.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}@${data.aws_ecr_image.api.image_digest}"
  architectures = ["x86_64"]
  # 2048 MB also doubles the CPU share, which roughly halves cold-start imports (FastAPI, the
  # Agents SDK, OpenAI, Chroma). At this traffic the difference is cents.
  memory_size = 2048
  # The agent can take a while (several tool calls, retries on rate limits); 120s matches the
  # old App Runner request limit.
  timeout = 120
  # The cost ceiling that API Gateway throttling gives the other apps: at most 10 chats at once.
  # Daily caps per visitor and overall live in the app (backend/src/ratelimit.py).
  reserved_concurrent_executions = 10

  environment {
    variables = {
      TABLE_NAME           = aws_dynamodb_table.main.name
      SSM_PREFIX           = local.ssm_prefix
      ORIGIN_SECRET        = random_password.origin_secret.result
      CHATS_PER_IP_PER_DAY = "50"
      CHATS_PER_DAY        = "1000"
      # Lambda Web Adapter: wait for uvicorn's port (the HTTP check would need the origin header)
      # and stream the response through instead of buffering it.
      AWS_LWA_READINESS_CHECK_PROTOCOL = "tcp"
      AWS_LWA_INVOKE_MODE              = "response_stream"
    }
  }

  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.api.name
  }
}

resource "aws_lambda_function_url" "api" {
  function_name = aws_lambda_function.api.function_name
  # Public like an API Gateway endpoint; the X-Origin-Verify check in the app keeps out anything
  # that didn't come through CloudFront. (CloudFront's Lambda OAC would need browsers to sign
  # every POST body with its SHA-256, so it doesn't fit a chat API.)
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"
}

# A public Function URL needs two permissions since October 2025. The provider adds the first
# (lambda:InvokeFunctionUrl, "FunctionURLAllowPublicAccess") itself when it creates the URL;
# this is the second. It waits for the URL: Lambda refuses two permission changes at once.
resource "aws_lambda_permission" "url_invoke" {
  statement_id             = "AllowPublicFunctionUrlInvoke"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.api.function_name
  principal                = "*"
  invoked_via_function_url = true

  depends_on = [aws_lambda_function_url.api]
}
