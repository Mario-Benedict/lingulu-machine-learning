"""
Middleware modules initialization.
"""
from app.middleware.metrics import setup_metrics
from app.middleware.auth import setup_auth_middleware, AuthenticationMiddleware

__all__ = ['setup_metrics', 'setup_auth_middleware', 'AuthenticationMiddleware']
