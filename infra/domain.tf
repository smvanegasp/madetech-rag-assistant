# The app's own address. DNS lives at Hostinger (sergiovanegas.com), so the two records below
# are added there by hand; nothing here touches Hostinger.
#   1. ACM's validation CNAME (output certificate_validation_record) proves the domain is ours.
#   2. nexus -> the CloudFront domain (output dns_target) sends visitors to the app.
# The cloudfront.net address keeps working too.

variable "domain" {
  description = "The app's address."
  type        = string
  default     = "nexus.sergiovanegas.com"
}

# CloudFront only uses certificates from us-east-1, which is this stack's region.
resource "aws_acm_certificate" "app" {
  domain_name       = var.domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

# Waits until ACM sees the validation record at Hostinger (usually minutes after adding it).
resource "aws_acm_certificate_validation" "app" {
  certificate_arn = aws_acm_certificate.app.arn

  timeouts {
    create = "2h"
  }
}

output "certificate_validation_record" {
  description = "Add this CNAME at Hostinger (Name without the trailing .sergiovanegas.com.)."
  value = [for o in aws_acm_certificate.app.domain_validation_options : {
    name  = o.resource_record_name
    type  = o.resource_record_type
    value = o.resource_record_value
  }]
}

output "dns_target" {
  description = "At Hostinger: CNAME nexus -> this."
  value       = aws_cloudfront_distribution.main.domain_name
}
