"""
Lingulu Machine Learning API
A production-ready Flask application for pronunciation prediction using Wav2Vec2.
"""
from flask import Flask

from app.config import get_config
from app.utils.logger import setup_logger, get_logger
from app.models import Wav2Vec2PronunciationModel
from app.utils import AudioProcessor
from app.middleware import setup_metrics, setup_auth_middleware
from app.routes import create_health_routes, create_prediction_routes


def create_app() -> Flask:
    """
    Application factory for creating Flask app.
    
    Returns:
        Configured Flask application
    """
    # Load configuration
    config = get_config()
    
    # Setup logging
    setup_logger('app', level=config.LOG_LEVEL, log_format=config.LOG_FORMAT)
    logger = get_logger(__name__)
    
    logger.info("Starting Lingulu ML Application")
    logger.info(f"Environment: {config.__name__}")
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Configure max content length for file uploads
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE_BYTES
    
    # Setup Prometheus metrics
    metrics, model_latency = setup_metrics(app, buckets=config.METRICS_BUCKETS)
    logger.info("Metrics configured successfully")
    
    # Setup authentication middleware
    auth_middleware = setup_auth_middleware(
        auth_service_url=config.AUTH_SERVICE_URL,
        timeout=config.AUTH_TIMEOUT
    )
    logger.info("Authentication middleware configured successfully")
    
    # Initialize components
    logger.info("Initializing ML model...")
    model = Wav2Vec2PronunciationModel(
        model_id=config.MODEL_ID,
        sampling_rate=config.SAMPLING_RATE
    )
    
    # Load model
    model.load()
    
    # Initialize audio processor
    audio_processor = AudioProcessor(
        sampling_rate=config.SAMPLING_RATE,
        max_file_size_bytes=config.MAX_FILE_SIZE_BYTES,
        max_audio_length_seconds=config.MAX_AUDIO_LENGTH_SECONDS,
        allowed_extensions=config.ALLOWED_EXTENSIONS
    )
    
    # Register blueprints
    logger.info("Registering routes...")
    
    health_routes = create_health_routes(model)
    app.register_blueprint(health_routes)
    
    prediction_routes = create_prediction_routes(model, audio_processor, model_latency, auth_middleware)
    app.register_blueprint(prediction_routes)
    
    logger.info("Application initialized successfully")
    
    return app


def main():
    """Main entry point for running the application."""
    config = get_config()
    app = create_app()
    
    logger = get_logger(__name__)
    logger.info(f"Starting server on {config.FLASK_HOST}:{config.FLASK_PORT}")
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )


if __name__ == '__main__':
    main()