"""
Health and readiness check endpoints.
"""
from flask import Blueprint, jsonify
from typing import Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)

health_bp = Blueprint('health', __name__, url_prefix='/api/model')


def create_health_routes(model):
    """
    Create health check routes with model dependency.
    
    Args:
        model: The ML model instance to check
        
    Returns:
        Blueprint with registered routes
    """
    
    @health_bp.route('/health', methods=['GET'])
    def health_check() -> tuple:
        """
        Basic health check endpoint.
        Returns 200 if service is running.
        """
        return jsonify({
            "status": "healthy",
            "service": "lingulu-ml"
        }), 200
    
    @health_bp.route('/readiness', methods=['GET'])
    def readiness_check() -> tuple:
        """
        Readiness check endpoint.
        Returns 200 only if model is loaded and ready.
        """
        if model.is_loaded():
            return jsonify({
                "status": "ready",
                "model": model.get_model_info()
            }), 200
        else:
            return jsonify({
                "status": "not_ready",
                "message": "Model not loaded"
            }), 503
    
    @health_bp.route('/model/info', methods=['GET'])
    def model_info() -> tuple:
        """
        Get model information endpoint.
        """
        info = model.get_model_info()
        if info.get("loaded", False):
            return jsonify(info), 200
        else:
            return jsonify({
                "error": "Model not loaded"
            }), 503
    
    return health_bp
