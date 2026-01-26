"""
Example: Testing the Pronunciation Assessment Model

This script demonstrates how to:
1. Load the trained model
2. Process audio files
3. Get phoneme predictions
4. Compare with expected pronunciation
"""

import torch
import librosa
from model.model import create_processor, get_device
from model.config import Config
from model.data_loader import text_to_ipa_phonemes


def load_trained_model(model_path: str):
    """Load trained pronunciation assessment model."""
    from model.model import Wav2Vec2ForPronunciationAssessment
    
    config = Config()
    device = get_device()
    
    # Create processor
    processor = create_processor(config.vocab_path, config.sampling_rate)
    
    # Load model
    model = Wav2Vec2ForPronunciationAssessment(
        config=config,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id
    )
    
    # Load trained weights
    checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model, processor, device


def process_audio(audio_path: str, processor, sampling_rate: int = 16000):
    """Process audio file for model input."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
    
    # Process with feature extractor
    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )
    
    return inputs


def predict_phonemes(model, processor, inputs, device):
    """Predict phonemes from audio input."""
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs["logits"]
        
        # Get predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_phonemes = processor.tokenizer.batch_decode(predicted_ids)
    
    return predicted_phonemes[0]


def compare_pronunciation(reference_text: str, predicted_phonemes: str, config):
    """Compare predicted phonemes with expected pronunciation."""
    # Get expected IPA phonemes from text
    expected_phonemes = text_to_ipa_phonemes(
        reference_text,
        include_stress=config.include_stress_markers,
        include_syllables=config.include_syllable_boundaries
    )
    
    # Calculate similarity (simple character-level)
    from jiwer import cer
    error_rate = cer(expected_phonemes, predicted_phonemes)
    accuracy = 1.0 - error_rate
    
    return {
        "expected": expected_phonemes,
        "predicted": predicted_phonemes,
        "accuracy": accuracy,
        "error_rate": error_rate
    }


def main():
    """Example usage of pronunciation assessment."""
    config = Config()
    
    # Paths
    model_path = "./wav2vec2-pronunciation-ctc/final"
    audio_path = "path/to/your/audio.wav"  # Replace with actual path
    reference_text = "hello world"  # The text that should be spoken
    
    print("=== Pronunciation Assessment Example ===\n")
    
    # 1. Load model
    print("Loading model...")
    model, processor, device = load_trained_model(model_path)
    print(f"✓ Model loaded on {device}\n")
    
    # 2. Process audio
    print(f"Processing audio: {audio_path}")
    inputs = process_audio(audio_path, processor, config.sampling_rate)
    print(f"✓ Audio processed\n")
    
    # 3. Predict phonemes
    print("Predicting phonemes...")
    predicted_phonemes = predict_phonemes(model, processor, inputs, device)
    print(f"✓ Predicted phonemes: {predicted_phonemes}\n")
    
    # 4. Compare with expected
    print(f"Reference text: '{reference_text}'")
    comparison = compare_pronunciation(reference_text, predicted_phonemes, config)
    
    print(f"\n=== Results ===")
    print(f"Expected:  {comparison['expected']}")
    print(f"Predicted: {comparison['predicted']}")
    print(f"Accuracy:  {comparison['accuracy']:.2%}")
    print(f"Error:     {comparison['error_rate']:.2%}")
    
    # Interpretation
    if comparison['accuracy'] >= 0.9:
        print("\n✓ Excellent pronunciation!")
    elif comparison['accuracy'] >= 0.7:
        print("\n○ Good pronunciation, minor errors")
    else:
        print("\n✗ Needs improvement")


if __name__ == "__main__":
    main()
