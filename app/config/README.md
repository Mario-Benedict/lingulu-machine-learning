# Configuration Module

This folder contains all configuration settings for the Lingulu ML application.

## Structure

```
config/
├── __init__.py      # Package initialization and exports
└── settings.py      # Configuration classes and environment settings
```

## Configuration Classes

- **Config**: Base configuration class with default values
- **DevelopmentConfig**: Development environment settings (DEBUG=True)
- **ProductionConfig**: Production environment settings (DEBUG=False)
- **TestConfig**: Test environment settings

## Environment Variables

Configure the application using these environment variables:

### Flask Settings
- `FLASK_HOST` - Server host (default: 0.0.0.0)
- `FLASK_PORT` - Server port (default: 5000)
- `FLASK_DEBUG` - Debug mode (default: False)
- `FLASK_ENV` - Environment: development, production, test (default: production)

### Model Settings
- `MODEL_ID` - HuggingFace model ID (default: marx90/lingulu_wav2vec2_pronounciation_finetune)
- `SAMPLING_RATE` - Audio sampling rate (default: 16000)

### Performance Settings
- `MAX_AUDIO_LENGTH_SECONDS` - Maximum audio duration (default: 60)
- `MAX_FILE_SIZE_MB` - Maximum upload size in MB (default: 10)

### Authentication Settings
- `AUTH_SERVICE_URL` - Spring Boot auth service URL (default: http://localhost:8080)
- `AUTH_TIMEOUT` - Auth request timeout in seconds (default: 5.0)

### Logging Settings
- `LOG_LEVEL` - Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

## Usage

```python
from app.config import get_config

# Get configuration based on FLASK_ENV
config = get_config()

# Access settings
print(config.MODEL_ID)
print(config.AUTH_SERVICE_URL)
```

## Adding New Settings

To add new configuration settings:

1. Add the setting to the `Config` class in `settings.py`
2. Use environment variables with `os.getenv()`
3. Add validation in `Config.validate()` if needed
4. Document the new setting in this README
