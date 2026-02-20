"""
Route modules initialization.
"""
from routes.health import create_health_routes
from routes.predict import create_prediction_routes
from routes.metrics import create_metrics_routes

__all__ = ['create_health_routes', 'create_prediction_routes', 'create_metrics_routes']
