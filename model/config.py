from dataclasses import dataclass

@dataclass
class Config:
    # Data
    sampling_rate: int = 16000
    
    # Model
    base_model: str = "facebook/wav2vec2-base-960h"
    vocab_path: str = "vocab.json"
    
    # Training
    output_dir: str = "./wav2vec2-phoneme-ctc"
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    fp16: bool = True
    logging_steps: int = 25
    save_total_limit: int = 2
    eval_steps: int = 500
    save_steps: int = 500
