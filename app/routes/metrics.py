"""
Metrics endpoint for monitoring API performance.
Provides p50, p90, p99 latency metrics.
"""
from flask import Blueprint, jsonify

from utils.metrics import get_metrics_tracker
from utils.logger import get_logger

logger = get_logger(__name__)

metrics_bp = Blueprint('metrics', __name__, url_prefix='/api')


def create_metrics_routes():
    """
    Create metrics monitoring routes.
    
    Returns:
        Blueprint with registered routes
    """
    
    @metrics_bp.route('/metrics', methods=['GET'])
    def get_metrics():
        """
        Get current API metrics including latency percentiles.
        
        Returns:
            JSON with p50, p90, p99 latency and other performance metrics
        """
        tracker = get_metrics_tracker()
        metrics = tracker.get_metrics()
        
        logger.debug("Metrics requested")
        
        return jsonify({
            "status": "success",
            "metrics": metrics
        }), 200
    
    @metrics_bp.route('/metrics/reset', methods=['POST'])
    def reset_metrics():
        """
        Reset all metrics (for testing/debugging).
        
        Returns:
            JSON confirmation
        """
        tracker = get_metrics_tracker()
        tracker.reset_metrics()
        
        logger.info("Metrics reset")
        
        return jsonify({
            "status": "success",
            "message": "Metrics have been reset"
        }), 200
    
    return metrics_bp
