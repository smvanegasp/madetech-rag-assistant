terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.69"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }

  # Created by infra/bootstrap. Native S3 lockfile, no DynamoDB lock table.
  # No profile here: GitHub Actions applies this stack with OIDC credentials, and locally the
  # credentials come from AWS_PROFILE=nexus (scripts/deploy.sh sets it).
  backend "s3" {
    bucket       = "nexus-tfstate-816940506502"
    key          = "nexus/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

# Same AWS account as pantry, fit and carhaul, fully separate stack: everything here is nexus-*.
provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = { app = "nexus" }
  }
}

data "aws_caller_identity" "current" {}

locals {
  name       = "nexus"
  account_id = data.aws_caller_identity.current.account_id
  # Secrets live here as SecureStrings, put there by scripts/set_aws_secrets.py (never Terraform).
  ssm_prefix = "/nexus/"
}
