output "url" {
  value = "https://${aws_cloudfront_distribution.main.domain_name}"
}

output "distribution_id" {
  value = aws_cloudfront_distribution.main.id
}

output "web_bucket" {
  value = aws_s3_bucket.web.bucket
}

output "ecr_repository" {
  value = aws_ecr_repository.api.repository_url
}

output "function_name" {
  value = aws_lambda_function.api.function_name
}

output "table_name" {
  value = aws_dynamodb_table.main.name
}

output "function_url" {
  description = "Direct calls are refused (403); the app is reached through CloudFront."
  value       = aws_lambda_function_url.api.function_url
}

output "site_url" {
  description = "The app's public address (the deploy workflow's smoke test uses it)."
  value       = "https://${var.domain}"
}
