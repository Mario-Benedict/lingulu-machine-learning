"""
Middleware modules initialization.
"""
from app.middleware.cloudwatch_metrics import setup_cloudwatch_metrics, CloudWatchMetrics

__all__ = ['setup_cloudwatch_metrics', 'CloudWatchMetrics']
