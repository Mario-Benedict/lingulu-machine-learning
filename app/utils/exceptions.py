"""
Custom exceptions for the Lingulu ML application.
"""


class LinguluMLException(Exception):
    """Base exception class for Lingulu ML application."""
    
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AudioProcessingError(LinguluMLException):
    """Exception raised when audio processing fails."""
    
    def __init__(self, message: str = "Failed to process audio file"):
        super().__init__(message, status_code=400)


class InvalidAudioFormatError(LinguluMLException):
    """Exception raised when audio format is invalid."""
    
    def __init__(self, message: str = "Invalid audio format"):
        super().__init__(message, status_code=400)


class FileTooLargeError(LinguluMLException):
    """Exception raised when uploaded file is too large."""
    
    def __init__(self, message: str = "File size exceeds maximum allowed size"):
        super().__init__(message, status_code=413)


class AudioTooLongError(LinguluMLException):
    """Exception raised when audio duration exceeds limit."""
    
    def __init__(self, message: str = "Audio duration exceeds maximum allowed length"):
        super().__init__(message, status_code=400)


class ModelInferenceError(LinguluMLException):
    """Exception raised when model inference fails."""
    
    def __init__(self, message: str = "Model inference failed"):
        super().__init__(message, status_code=500)


class ModelNotLoadedError(LinguluMLException):
    """Exception raised when model is not properly loaded."""
    
    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message, status_code=503)


class InvalidRequestError(LinguluMLException):
    """Exception raised when request is invalid."""
    
    def __init__(self, message: str = "Invalid request"):
        super().__init__(message, status_code=400)
