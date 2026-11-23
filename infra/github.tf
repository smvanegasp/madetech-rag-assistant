# Lets .github/workflows/deploy.yml deploy code with short-lived credentials (OIDC), no AWS keys
# in GitHub. The role can push the image, roll Lambda, upload the frontend and invalidate
# CloudFront; infrastructure changes still go through a local `terraform apply`.

locals {
  # The repo sends GitHub's immutable subject ("owner@ownerId/repo@repoId"), so a deleted and
  # re-created repo with the same name can't match. Check the format with:
  #   gh api repos/smvanegasp/madetech-rag-assistant/actions/oidc/customization/sub
  # (turned on with: gh api -X PUT repos/smvanegasp/madetech-rag-assistant/actions/oidc/customization/sub
  #   -F use_default=true -F use_immutable_subject=true)
  github_repo = "smvanegasp@145235327/madetech-rag-assistant@1169614001"
}

# One per AWS account, and the pantry stack (same account) created it: read it, don't create it.
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

# Merges deploy everything, infrastructure included (.github/workflows/deploy.yml runs
# `terraform apply`), so this role needs to manage any resource in the stack: IAM roles,
# Lambda, CloudFront, DynamoDB... It gets AdministratorAccess, and only workflows running on
# main (merges, which only Sergio can make) can assume it; never PRs or other branches.
resource "aws_iam_role" "github_deploy" {
  name = "${local.name}-github-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = data.aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          "token.actions.githubusercontent.com:sub" = "repo:${local.github_repo}:ref:refs/heads/main"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "github_deploy" {
  role       = aws_iam_role.github_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

# PRs show what a merge would change in AWS (`terraform plan` in .github/workflows/ci.yml).
# Read-only, and the plan runs with -lock=false so it never writes to the state bucket. GitHub
# doesn't give OIDC tokens to PRs from forks or Dependabot, so only Sergio's own branches can use it.
resource "aws_iam_role" "github_plan" {
  name = "${local.name}-github-plan"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = data.aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          "token.actions.githubusercontent.com:sub" = "repo:${local.github_repo}:pull_request"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "github_plan" {
  role       = aws_iam_role.github_plan.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

output "github_deploy_role" {
  value = aws_iam_role.github_deploy.arn
}

output "github_plan_role" {
  value = aws_iam_role.github_plan.arn
}
