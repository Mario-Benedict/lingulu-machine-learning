"""
Route modules initialization.
"""
from app.routes.health import create_health_routes
from app.routes.predict import create_prediction_routes
from app.routes.metrics import create_metrics_routes

__all__ = ['create_health_routes', 'create_prediction_routes', 'create_metrics_routes']
