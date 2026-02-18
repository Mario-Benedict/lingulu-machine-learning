import unittest
from app.utils.audio_processor import AudioProcessor
from app.utils.gop_calculator import PhonemeScore, WordScore, SentenceScore, GOPCalculator
from app.utils.phoneme_converter import PhonemeConverter
from app.utils.logger import setup_logger, get_logger
from app.utils.exceptions import (
    LinguluMLException, AudioProcessingError, InvalidAudioFormatError, FileTooLargeError,
    AudioTooLongError, ModelInferenceError, ModelNotLoadedError, InvalidRequestError
)
from app.models.wav2vec2_model import Wav2Vec2PronunciationModel
from unittest.mock import Mock, patch
import numpy as np
import soundfile as sf
import io

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor()

    def test_validate_file_extension(self):
        self.processor.validate_file_extension('test.wav')
        with self.assertRaises(InvalidAudioFormatError):
            self.processor.validate_file_extension('test.txt')


    def test_validate_file_size(self):
        # Mock FileStorage with seek and tell
        mock_file = Mock()
        # Simulate a small file
        mock_file.seek = Mock()
        mock_file.tell = Mock(return_value=1000)
        self.processor.validate_file_size(mock_file)
        # Simulate a large file
        mock_file.tell = Mock(return_value=100000000)
        with self.assertRaises(FileTooLargeError):
            self.processor.validate_file_size(mock_file)

    def test_load_audio(self):
        # Mock FileStorage to return valid audio bytes
        mock_file = Mock()
        # Generate a short valid wav file in memory using numpy and librosa
        arr = np.random.randn(16000)
        buf = io.BytesIO()
        sf.write(buf, arr, 16000, format='WAV')
        buf.seek(0)
        mock_file.read = Mock(return_value=buf.read())
        mock_file.filename = 'test.wav'
        # Now test load_audio
        result_arr, sr = self.processor.load_audio(mock_file)
        self.assertEqual(sr, 16000)
        self.assertTrue(isinstance(result_arr, np.ndarray))

class TestGOPCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = GOPCalculator()

    def test_normalize_gop_to_percentage(self):
        scores = [0.5, 0.75, 1.0]
        result = self.calculator.normalize_gop_to_percentage(scores)
        self.assertTrue(all(0 <= x <= 100 for x in result))

class TestPhonemeConverter(unittest.TestCase):
    def setUp(self):
        self.converter = PhonemeConverter()

    def test_arpabet_to_ipa(self):
        arpabet = ['AA', 'B']
        ipa = self.converter.arpabet_to_ipa(arpabet)
        self.assertIsInstance(ipa, list)

class TestLogger(unittest.TestCase):
    def test_setup_logger(self):
        logger = setup_logger('test')
        self.assertIsNotNone(logger)

    def test_get_logger(self):
        logger = get_logger('test')
        self.assertIsNotNone(logger)

class TestExceptions(unittest.TestCase):
    def test_lingulu_ml_exception(self):
        e = LinguluMLException('msg', 400)
        self.assertEqual(e.status_code, 400)

    def test_audio_processing_error(self):
        e = AudioProcessingError()
        self.assertIsInstance(e, LinguluMLException)

    def test_invalid_audio_format_error(self):
        e = InvalidAudioFormatError()
        self.assertIsInstance(e, LinguluMLException)

    def test_file_too_large_error(self):
        e = FileTooLargeError()
        self.assertIsInstance(e, LinguluMLException)

    def test_audio_too_long_error(self):
        e = AudioTooLongError()
        self.assertIsInstance(e, LinguluMLException)

    def test_model_inference_error(self):
        e = ModelInferenceError()
        self.assertIsInstance(e, LinguluMLException)

    def test_model_not_loaded_error(self):
        e = ModelNotLoadedError()
        self.assertIsInstance(e, LinguluMLException)

    def test_invalid_request_error(self):
        e = InvalidRequestError()
        self.assertIsInstance(e, LinguluMLException)

class TestWav2Vec2PronunciationModel(unittest.TestCase):
    def setUp(self):
        model_id = "marx90/lingulu_wav2vec2_pronounciation_finetune"
        self.model = Wav2Vec2PronunciationModel(model_id)

    def test_is_loaded(self):
        self.assertIsInstance(self.model.is_loaded(), bool)

    def test_get_model_info(self):
        info = self.model.get_model_info()
        self.assertIsInstance(info, dict)

if __name__ == '__main__':
    unittest.main()
