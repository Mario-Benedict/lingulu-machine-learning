import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Model
)


class Wav2Vec2ForPronunciationAssessment(nn.Module):
    """
    Custom Wav2Vec2 model with additional layers for pronunciation assessment.
    Combines pretrained Wav2Vec2 with custom pronunciation scoring head.
    """
    def __init__(self, config, vocab_size, pad_token_id):
        super().__init__()
        
        # Load pretrained Wav2Vec2 model (encoder only)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            config.base_model,
            ignore_mismatched_sizes=True
        )
        
        # Freeze feature extractor if needed
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Get hidden size from pretrained model
        self.hidden_size = self.wav2vec2.config.hidden_size
        
        # Additional pronunciation-specific layers
        if config.use_custom_head:
            self.pronunciation_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=8,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.hidden_dropout,
                    batch_first=True
                )
                for _ in range(config.num_pronunciation_layers)
            ])
            
            # Pronunciation scoring head
            self.pronunciation_head = nn.Sequential(
                nn.Linear(self.hidden_size, config.pronunciation_head_dim),
                nn.GELU(),
                nn.Dropout(config.hidden_dropout),
                nn.Linear(config.pronunciation_head_dim, config.pronunciation_head_dim // 2),
                nn.GELU(),
                nn.Dropout(config.hidden_dropout),
            )
        else:
            self.pronunciation_layers = None
            self.pronunciation_head = None
        
        # CTC head for phoneme recognition
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.lm_head = nn.Linear(
            config.pronunciation_head_dim // 2 if config.use_custom_head else self.hidden_size,
            vocab_size
        )
        
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
    def freeze_feature_extractor(self):
        """Freeze the feature extractor parameters."""
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
        return_dict=True
    ):
        # Extract features from pretrained model
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Apply custom pronunciation layers
        if self.pronunciation_layers is not None:
            for layer in self.pronunciation_layers:
                hidden_states = layer(hidden_states)
            
            # Apply pronunciation head
            hidden_states = self.pronunciation_head(hidden_states)
        
        # Apply dropout and final projection
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # CTC Loss
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            
            # Get input lengths
            if attention_mask is not None:
                input_lengths = attention_mask.sum(-1)
            else:
                input_lengths = torch.full(
                    (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
                )
            
            # Get label lengths (excluding padding)
            labels_mask = labels != self.pad_token_id
            target_lengths = labels_mask.sum(-1)
            
            # Compute CTC loss
            loss = nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=self.pad_token_id,
                reduction="mean",
                zero_infinity=True
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
        }


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
        return_attention_mask=True  # Enable attention mask for custom model
    )
    
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    return processor


def create_model(config, processor, device):
    """Create and configure custom Wav2Vec2 model for pronunciation assessment."""
    model = Wav2Vec2ForPronunciationAssessment(
        config=config,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id
    )
    
    model.freeze_feature_extractor()
    model = model.to(device)
    
    print(f"\n=== Model Architecture ===")
    print(f"Base model: {config.base_model}")
    print(f"Custom pronunciation layers: {config.num_pronunciation_layers}")
    print(f"Pronunciation head dim: {config.pronunciation_head_dim}")
    print(f"Vocab size: {len(processor.tokenizer)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

