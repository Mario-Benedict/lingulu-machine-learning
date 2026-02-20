"""
Wav2Vec2 model wrapper for pronunciation prediction.
Handles model loading, inference, and result processing.
"""
import time
import torch
import numpy as np
from typing import Dict, Any, Optional
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.utils.logger import get_logger
from app.utils.exceptions import ModelInferenceError, ModelNotLoadedError
from app.utils.gop_calculator import GOPCalculator

logger = get_logger(__name__)


class Wav2Vec2PronunciationModel:
    """
    Wrapper for Wav2Vec2 pronunciation model.
    Handles model loading, caching, and inference.
    """
    
    def __init__(
        self,
        model_id: str,
        sampling_rate: int = 16000,
        enable_gop: bool = True
    ):
        """
        Initialize the Wav2Vec2 model wrapper.
        
        Args:
            model_id: Hugging Face model identifier
            sampling_rate: Expected audio sampling rate
            enable_gop: Enable GOP (Goodness of Pronunciation) scoring
        """
        self.model_id = model_id
        self.sampling_rate = sampling_rate
        self.enable_gop = enable_gop
        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[Wav2Vec2ForCTC] = None
        self._is_loaded = False
        
        # Device configuration - use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GOP calculator (will be initialized with g2p on first use)
        self.gop_calculator: Optional[GOPCalculator] = None
        self.g2p_model = None
        
        # Log device info prominently
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU DETECTED: {gpu_name} with {gpu_memory:.1f}GB memory")
            logger.info(f"✅ Using device: {self.device} (CUDA acceleration enabled)")
        else:
            logger.warning(f"⚠️  No GPU detected - Using device: {self.device} (CPU only)")
        
        logger.info(
            f"Initializing Wav2Vec2PronunciationModel with model: {model_id}, "
            f"GOP enabled: {enable_gop}"
        )
    
    def load(self) -> None:
        """
        Load the model and processor from Hugging Face.
        
        Raises:
            ModelInferenceError: If model loading fails
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping...")
            return
        
        try:
            # Initialize GOP calculator if enabled
            if self.enable_gop:
                gop_init_success = False
                try:
                    from g2p_en import G2p
                    logger.debug("Loading G2P model for GOP calculation...")
                    self.g2p_model = G2p()
                    self.gop_calculator = GOPCalculator()
                    gop_init_success = True
                    logger.info("✅ GOP calculator initialized successfully")
                except ImportError as e:
                    logger.warning(
                        f"g2p-en not installed. GOP scoring disabled. "
                        f"Install with: pip install g2p-en. Error: {e}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize GOP calculator: {e}", exc_info=True)
                
                # Ensure enable_gop accurately reflects GOP availability
                if not gop_init_success:
                    self.enable_gop = False
                    self.gop_calculator = None
                    self.g2p_model = None
                    logger.info("GOP scoring is DISABLED for this instance")
                else:
                    logger.info("GOP scoring is ENABLED and ready")
            
            logger.info(f"Loading model from Hugging Face: {self.model_id}")
            start_time = time.time()
            
            # Load processor
            logger.debug("Loading processor...")
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            
            # Load model
            logger.debug("Loading model...")
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            
            # Move model to device (GPU if available)
            logger.debug(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable inference optimization
            torch.set_grad_enabled(False)
            if hasattr(torch, 'inference_mode'):
                logger.debug("Enabling inference_mode for faster inference")
            
            load_time = time.time() - start_time
            self._is_loaded = True
            
            logger.info(
                f"Model loaded successfully in {load_time:.2f}s on {self.device}. "
                f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )
            
            # Log GPU memory usage if using CUDA
            if self.device.type == 'cuda':
                logger.info(
                    f"GPU Memory - Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB, "
                    f"Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB"
                )
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise ModelInferenceError(f"Failed to load model: {str(e)}")
    
    def predict(
        self,
        audio_array: np.ndarray,
        reference_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run inference on audio array.
        
        Args:
            audio_array: Audio data as numpy array
            reference_text: Optional reference text for GOP calculation
            
        Returns:
            Dictionary containing transcription and metadata
            
        Raises:
            ModelNotLoadedError: If model is not loaded
            ModelInferenceError: If inference fails
        """
        if not self._is_loaded or self.model is None or self.processor is None:
            raise ModelNotLoadedError("Model must be loaded before prediction")
        
        try:
            logger.debug(f"Running inference on audio with {len(audio_array)} samples")
            start_time = time.perf_counter()
            
            # Preprocessing with optimized settings
            inputs = self.processor(
                audio_array,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device (non-blocking for speed on GPU)
            input_values = inputs.input_values.to(self.device, non_blocking=True)
            
            # Inference with GPU optimization and mixed precision
            with torch.no_grad():
                if self.device.type == 'cuda':
                    # Use automatic mixed precision for 2x speedup on GPU
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(input_values).logits
                else:
                    logits = self.model(input_values).logits
            
            # Decoding - delay CPU transfer
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids.cpu())[0]
            
            latency = time.perf_counter() - start_time
            
            result = {
                "transcription": transcription,
                "latency_seconds": round(latency, 5),
                "audio_samples": len(audio_array),
                "audio_duration_seconds": round(len(audio_array) / self.sampling_rate, 2)
            }
            
            # Calculate GOP if reference text provided and GOP enabled
            if reference_text:
                # Check if GOP is fully initialized and enabled
                if self.enable_gop and self.gop_calculator is not None and self.g2p_model is not None:
                    try:
                        logger.debug(f"Calculating GOP for reference: '{reference_text}'")
                        gop_start = time.perf_counter()
                        
                        # Get log probabilities for GOP - keep on GPU for speed
                        log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
                        
                        # Calculate GOP scores with GPU acceleration
                        sentence_score = self.gop_calculator.calculate_sentence_score(
                            text=reference_text,
                            log_probs=log_probs,
                            tokenizer=self.processor.tokenizer,
                            g2p_model=self.g2p_model,
                            device=self.device
                        )
                        
                        gop_latency = time.perf_counter() - gop_start
                        
                        # Add GOP results to output
                        result['pronounciation_assessment'] = sentence_score.to_dict()
                        result['gop_latency_seconds'] = round(gop_latency, 4)
                        
                        logger.info(
                            f"GOP calculated: avg score {sentence_score.average_score:.1f}% "
                            f"in {gop_latency:.4f}s"
                        )
                        
                    except Exception as e:
                        logger.error(f"GOP calculation failed: {e}", exc_info=True)
                        result['pronounciation_assessment'] = {
                            'error': 'GOP calculation failed',
                            'details': str(e)
                        }
                else:
                    # GOP requested but not available
                    logger.debug(f"GOP requested but not available (enable_gop={self.enable_gop}, gop_calculator={self.gop_calculator is not None}, g2p_model={self.g2p_model is not None})")
                    result['pronounciation_assessment'] = {
                        'error': 'GOP not enabled. Install g2p-en to enable GOP scoring.',
                        'gop_status': 'disabled'
                    }
            
            logger.info(
                f"Inference completed in {latency:.4f}s. "
                f"Transcription length: {len(transcription)} chars"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}", exc_info=True)
            raise ModelInferenceError(f"Inference failed: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self._is_loaded:
            return {
                "loaded": False,
                "device": str(self.device)
            }
        
        model_info = {
            "loaded": True,
            "model_id": self.model_id,
            "sampling_rate": self.sampling_rate,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "gop_enabled": self.enable_gop,
            "device": str(self.device),
        }
        
        # Add GPU memory info if using CUDA
        if self.device.type == 'cuda':
            model_info.update({
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 2),
                "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 2),
                "gpu_name": torch.cuda.get_device_name(0),
            })
        
        return model_info
    
    def clear_cache(self) -> None:
        """Clear GPU cache if using CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
