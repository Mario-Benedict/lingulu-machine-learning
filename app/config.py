"""
Configuration module for the Lingulu ML application.
Manages all configuration settings with environment variable support.
"""
import os
from typing import List


class Config:
    """Base configuration class."""
    
    # Flask Configuration
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Model Configuration
    MODEL_ID = os.getenv('MODEL_ID', 'marx90/lingulu_wav2vec2_pronounciation_finetune')
    SAMPLING_RATE = int(os.getenv('SAMPLING_RATE', 16000))
    
    # Performance Configuration
    MAX_AUDIO_LENGTH_SECONDS = int(os.getenv('MAX_AUDIO_LENGTH_SECONDS', 60))
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 10))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Allowed audio file extensions
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
    
    # Prometheus Metrics Configuration
    METRICS_BUCKETS: List[float] = [1.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, float("inf")]
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @staticmethod
    def validate():
        """Validate configuration settings."""
        if Config.MAX_AUDIO_LENGTH_SECONDS <= 0:
            raise ValueError("MAX_AUDIO_LENGTH_SECONDS must be positive")
        if Config.MAX_FILE_SIZE_MB <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be positive")
        if Config.SAMPLING_RATE not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError(f"Invalid SAMPLING_RATE: {Config.SAMPLING_RATE}")


class DevelopmentConfig(Config):
    """Development environment configuration."""
    FLASK_DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration."""
    FLASK_DEBUG = False
    LOG_LEVEL = 'WARNING'


class TestConfig(Config):
    """Test environment configuration."""
    FLASK_DEBUG = True
    LOG_LEVEL = 'DEBUG'


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'production').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'test': TestConfig,
    }
    
    config_class = config_map.get(env, ProductionConfig)
    config_class.validate()
    return config_class
