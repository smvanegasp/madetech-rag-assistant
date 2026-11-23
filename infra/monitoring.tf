# Email alerts: the API crashing or throttled, chats failing, and the daily caps being hit.
# Free at this size (CloudWatch's first 10 alarms and metrics, SNS's first 1,000 emails).

variable "alert_email" {
  description = "Where alerts go. Locally infra/terraform.tfvars (git-ignored); in CI the ALERT_EMAIL repo variable."
  type        = string
  # Keeps the address out of plan output, which is public in this repo's Actions logs.
  sensitive = true
}

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

# AWS emails a confirmation link first; alerts only arrive after it's clicked.
resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

locals {
  # Stable log lines from the API (backend/src/app.py, ratelimit.py, chat_logger.py).
  log_signals = {
    StreamError      = "\"Stream error\""
    ChatError        = "\"Chat endpoint error\""
    ContactError     = "\"Contact endpoint error\""
    ChatLogFailed    = "\"Failed to log chat message\""
    GlobalCapReached = "\"Chat rate limit reached (global)\""
  }
}

resource "aws_cloudwatch_log_metric_filter" "signals" {
  for_each       = local.log_signals
  name           = each.key
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = each.value

  metric_transformation {
    name          = each.key
    namespace     = "Nexus"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_metric_alarm" "chat_failing" {
  alarm_name          = "${local.name}-chat-failing"
  alarm_description   = "Chats failed 3+ times in 15 minutes (Groq/OpenAI/Chroma down, a key revoked or a spend limit hit). Visitors see \"I'm having trouble right now\"."
  namespace           = "Nexus"
  metric_name         = "StreamError"
  statistic           = "Sum"
  period              = 900
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  depends_on          = [aws_cloudwatch_log_metric_filter.signals]
}

resource "aws_cloudwatch_metric_alarm" "contact_failing" {
  alarm_name          = "${local.name}-contact-failing"
  alarm_description   = "A contact/feedback email could not be sent (Resend key or CONTACT_EMAIL problem)."
  namespace           = "Nexus"
  metric_name         = "ContactError"
  statistic           = "Sum"
  period              = 900
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  depends_on          = [aws_cloudwatch_log_metric_filter.signals]
}

resource "aws_cloudwatch_metric_alarm" "global_cap" {
  alarm_name          = "${local.name}-daily-cap-reached"
  alarm_description   = "The global daily chat cap (CHATS_PER_DAY) was reached: visitors get the come-back-tomorrow message until midnight UTC."
  namespace           = "Nexus"
  metric_name         = "GlobalCapReached"
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  depends_on          = [aws_cloudwatch_log_metric_filter.signals]
}

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${local.name}-api-crashing"
  alarm_description   = "The API function crashed or timed out (for example it can't start: missing SSM secret, Chroma unreachable)."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  alarm_name          = "${local.name}-api-throttled"
  alarm_description   = "Requests were refused because all reserved concurrency (10) was busy: unusual traffic, or raise reserved_concurrent_executions."
  namespace           = "AWS/Lambda"
  metric_name         = "Throttles"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "lambda_slow" {
  alarm_name          = "${local.name}-api-slow"
  alarm_description   = "Typical (p90) request time stayed above 60s for 15 minutes: the model provider is slow or retrying."
  namespace           = "AWS/Lambda"
  metric_name         = "Duration"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  extended_statistic  = "p90"
  period              = 900
  evaluation_periods  = 1
  threshold           = 60000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
