import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)


def get_device():
    """Get the available device (cuda or cpu)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def create_processor(vocab_path: str, sampling_rate: int = 16000):
    """Create Wav2Vec2 processor with tokenizer and feature extractor."""
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        sampling_rate=sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )
    
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    return processor


def create_model(config, processor, device):
    """Create and configure Wav2Vec2ForCTC model."""
    model = Wav2Vec2ForCTC.from_pretrained(
        config.base_model,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ignore_mismatched_sizes=True
    )
    
    model.freeze_feature_extractor()
    model = model.to(device)
    
    return model
