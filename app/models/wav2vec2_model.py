"""
Wav2Vec2 model wrapper for pronunciation prediction.
Handles model loading, inference, and result processing.
"""
import os
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
            
            # Try to compile model with PyTorch 2.0+ for extra speedup
            # Can be disabled via DISABLE_TORCH_COMPILE=1 environment variable
            disable_compile = os.environ.get('DISABLE_TORCH_COMPILE', '0') == '1'
            
            if hasattr(torch, 'compile') and self.device.type == 'cuda' and not disable_compile:
                try:
                    logger.info("Attempting to compile model with torch.compile()...")
                    
                    # Configure torch._dynamo to suppress errors and fall back to eager mode
                    if hasattr(torch, '_dynamo'):
                        torch._dynamo.config.suppress_errors = True
                    
                    # Use 'default' mode instead of 'reduce-overhead' for better compatibility
                    compiled_model = torch.compile(self.model, mode='default')
                    
                    # Test compilation with dummy input before committing
                    logger.debug("Testing compiled model with dummy input...")
                    dummy_input = torch.randn(1, 16000, device=self.device)
                    with torch.inference_mode():
                        _ = compiled_model(dummy_input)
                    
                    # If test succeeded, use compiled model
                    self.model = compiled_model
                    logger.info("✅ Model compiled successfully with torch.compile()")
                    
                except Exception as e:
                    logger.warning(
                        f"torch.compile() failed or unavailable: {str(e)[:200]}. "
                        f"Falling back to eager mode (still optimized with inference_mode + mixed precision)."
                    )
                    # Ensure we're using the original uncompiled model
                    pass
            elif disable_compile:
                logger.info("torch.compile() disabled via DISABLE_TORCH_COMPILE environment variable")
            
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
            start_time = time.perf_counter()
            
            # Fast preprocessing - no logging overhead
            with torch.inference_mode():
                # Process audio directly
                inputs = self.processor(
                    audio_array,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=False
                )
                
                input_values = inputs.input_values.to(self.device, non_blocking=True)
                
                # Model inference with mixed precision on GPU
                if self.device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(input_values).logits
                else:
                    logits = self.model(input_values).logits
            
            # Decode transcription (happens once, outside inference_mode)
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids.cpu())[0]
            
            # GOP calculation if needed (separate context for clarity)
            gop_result = None
            if reference_text and self.enable_gop and self.gop_calculator and self.g2p_model:
                gop_start = time.perf_counter()
                try:
                    # Calculate log probs outside inference_mode (already computed)
                    log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
                    
                    sentence_score = self.gop_calculator.calculate_sentence_score(
                        text=reference_text,
                        log_probs=log_probs,
                        tokenizer=self.processor.tokenizer,
                        g2p_model=self.g2p_model,
                        device=self.device
                    )
                        
                    gop_latency = time.perf_counter() - gop_start
                    gop_result = sentence_score.to_dict()
                    gop_result['gop_latency_seconds'] = round(gop_latency, 5)
                except Exception as e:
                    logger.error(f"GOP failed: {str(e)[:100]}")
                    gop_result = {'error': 'GOP calculation failed', 'details': str(e)}
            
            latency = time.perf_counter() - start_time
            
            # Build result quickly
            result = {
                "transcription": transcription,
                "latency_seconds": round(latency, 5),
                "audio_samples": len(audio_array),
                "audio_duration_seconds": round(len(audio_array) / self.sampling_rate, 2)
            }
            
            # Add GOP if available
            if reference_text:
                if gop_result:
                    result['pronounciation_assessment'] = gop_result
                else:
                    result['pronounciation_assessment'] = {
                        'error': 'GOP not enabled',
                        'gop_status': 'disabled'
                    }
            
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
