"""
Middleware modules initialization.
"""
from app.middleware.metrics import setup_metrics

__all__ = ['setup_metrics']
