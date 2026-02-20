"""
Lingulu Machine Learning API
A production-ready Flask application for pronunciation prediction using Wav2Vec2.
"""
import os
from pathlib import Path
from flask import Flask

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from {env_file}")
except ImportError:
    # python-dotenv not installed, skip
    pass

from app.config import get_config
from app.utils.logger import setup_logger, get_logger
from app.models import Wav2Vec2PronunciationModel
from app.utils import AudioProcessor
from app.routes import create_health_routes, create_prediction_routes, create_metrics_routes


def create_app() -> Flask:
    """
    Application factory for creating Flask app.
    
    Returns:
        Configured Flask application
    """
    # Load configuration
    config = get_config()
    
    # Setup logging with stdout for container visibility
    import logging
    import sys
    from app.utils.logger import FlushingStreamHandler
    
    # Configure root logger for gunicorn compatibility
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Create flushing stdout handler for container logs
    console_handler = FlushingStreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    formatter = logging.Formatter(config.LOG_FORMAT)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Setup app logger
    setup_logger('app', level=config.LOG_LEVEL, log_format=config.LOG_FORMAT)
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("Starting Lingulu ML Application")
    logger.info(f"Environment: {os.getenv('FLASK_ENV', 'production')}")
    logger.info(f"Log Level: {config.LOG_LEVEL}")
    logger.info("="*60)
    
    # Force flush to ensure logs appear in container stdout immediately
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Configure max content length for file uploads
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE_BYTES
    
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
    
    prediction_routes = create_prediction_routes(model, audio_processor)
    app.register_blueprint(prediction_routes)
    
    metrics_routes = create_metrics_routes()
    app.register_blueprint(metrics_routes)
    
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