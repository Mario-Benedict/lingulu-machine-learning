"""
Audio processing utilities.
Handles audio file validation, loading, and preprocessing.
"""
import io
import librosa
import numpy as np
from typing import Tuple, BinaryIO
from werkzeug.datastructures import FileStorage

from app.utils.logger import get_logger
from app.utils.exceptions import (
    AudioProcessingError,
    InvalidAudioFormatError,
    FileTooLargeError,
    AudioTooLongError
)

logger = get_logger(__name__)


class AudioProcessor:
    """Handles audio file processing and validation."""
    
    def __init__(
        self,
        sampling_rate: int = 16000,
        max_file_size_bytes: int = 10 * 1024 * 1024,
        max_audio_length_seconds: int = 60,
        allowed_extensions: set | None = None
    ):
        """
        Initialize AudioProcessor.
        
        Args:
            sampling_rate: Target sampling rate for audio
            max_file_size_bytes: Maximum allowed file size in bytes
            max_audio_length_seconds: Maximum allowed audio duration in seconds
            allowed_extensions: Set of allowed file extensions
        """
        self.sampling_rate = sampling_rate
        self.max_file_size_bytes = max_file_size_bytes
        self.max_audio_length_seconds = max_audio_length_seconds
        self.allowed_extensions = allowed_extensions or {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
        
        logger.info(
            f"AudioProcessor initialized: sr={sampling_rate}, "
            f"max_size={max_file_size_bytes/1024/1024}MB, "
            f"max_duration={max_audio_length_seconds}s"
        )
    
    def validate_file_extension(self, filename: str) -> None:
        """
        Validate file extension.
        
        Args:
            filename: Name of the file
            
        Raises:
            InvalidAudioFormatError: If extension is not allowed
        """
        if '.' not in filename:
            raise InvalidAudioFormatError("File has no extension")
        
        extension = filename.rsplit('.', 1)[1].lower()
        if extension not in self.allowed_extensions:
            raise InvalidAudioFormatError(
                f"File extension '{extension}' not allowed. "
                f"Allowed: {', '.join(self.allowed_extensions)}"
            )
    
    def validate_file_size(self, file: FileStorage) -> None:
        """
        Validate file size.
        
        Args:
            file: Uploaded file
            
        Raises:
            FileTooLargeError: If file is too large
        """
        # Seek to end to get file size
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > self.max_file_size_bytes:
            raise FileTooLargeError(
                f"File size {file_size/1024/1024:.2f}MB exceeds "
                f"maximum {self.max_file_size_bytes/1024/1024}MB"
            )
        
        logger.debug(f"File size: {file_size/1024/1024:.2f}MB")
    
    def load_audio(self, audio_file: FileStorage) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_file: Uploaded audio file
            
        Returns:
            Tuple of (audio_array, sampling_rate)
            
        Raises:
            AudioProcessingError: If loading fails
            AudioTooLongError: If audio is too long
        """
        try:
            # Read file into bytes
            audio_bytes = io.BytesIO(audio_file.read())
            
            # Load with librosa
            logger.debug(f"Loading audio file: {audio_file.filename}")
            speech_array, sr = librosa.load(
                audio_bytes,
                sr=self.sampling_rate,
                mono=True
            )
            
            # Validate duration
            duration = len(speech_array) / sr
            logger.debug(f"Audio duration: {duration:.2f}s")
            
            if duration > self.max_audio_length_seconds:
                raise AudioTooLongError(
                    f"Audio duration {duration:.2f}s exceeds "
                    f"maximum {self.max_audio_length_seconds}s"
                )
            
            if len(speech_array) == 0:
                raise AudioProcessingError("Audio file is empty")
            
            logger.info(
                f"Successfully loaded audio: duration={duration:.2f}s, "
                f"samples={len(speech_array)}"
            )
            
            return speech_array, int(sr)
            
        except (AudioTooLongError, AudioProcessingError):
            raise
        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}", exc_info=True)
            raise AudioProcessingError(f"Failed to load audio file: {str(e)}")
    
    def validate_and_load(self, audio_file: FileStorage) -> Tuple[np.ndarray, int]:
        """
        Validate and load audio file in one step.
        
        Args:
            audio_file: Uploaded audio file
            
        Returns:
            Tuple of (audio_array, sampling_rate)
        """
        self.validate_file_extension(audio_file.filename)
        self.validate_file_size(audio_file)
        return self.load_audio(audio_file)
