"""
Prediction endpoint for audio transcription.
"""
from flask import Blueprint, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from app.utils.logger import get_logger
from app.utils.metrics import track_latency
from app.utils.exceptions import (
    LinguluMLException,
    InvalidRequestError
)

logger = get_logger(__name__)

predict_bp = Blueprint('predict', __name__, url_prefix='/api/model')


def create_prediction_routes(model, audio_processor):
    """
    Create prediction routes with dependencies.
    
    Args:
        model: The ML model instance
        audio_processor: AudioProcessor instance
        
    Returns:
        Blueprint with registered routes
    """
    
    @predict_bp.route('/predict', methods=['POST'])
    @track_latency
    def predict():
        """
        Predict pronunciation from audio file.
        
        Expected:
            - POST request with multipart/form-data
            - 'file' field containing audio file
            - 'text' field (optional) containing reference text for GOP scoring
            
        Returns:
            JSON with transcription and metadata (including GOP if text provided)
        """
        # Fast validation
        if 'file' not in request.files:
            raise InvalidRequestError("No file provided. Please upload an audio file in 'file' field")
        
        audio_file = request.files['file']
        
        if audio_file.filename == '':
            raise InvalidRequestError("No file selected")
        
        # Get optional reference text for GOP
        reference_text = request.form.get('text', '').strip()
        
        try:
            # Fast audio loading and processing
            speech_array, sampling_rate = audio_processor.validate_and_load(audio_file)
            
            # Run inference
            result = model.predict(
                speech_array,
                reference_text=reference_text if reference_text else None
            )
            
            # Build response quickly
            result['status'] = 'success'
            result['filename'] = audio_file.filename
            
            if reference_text:
                result['reference_text'] = reference_text
            
            return jsonify(result), 200
            
        except LinguluMLException as e:
            return jsonify({
                "error": e.message,
                "status": "error"
            }), e.status_code
            
        except RequestEntityTooLarge:
            return jsonify({
                "error": "File too large",
                "status": "error"
            }), 413
            
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Internal server error",
                "status": "error"
            }), 500
    
    return predict_bp
