# Data
sampling_rate: int = 16000

# Dataset settings
use_local_dataset: bool = True  # Use local LibriSpeech dataset
local_dataset_path: str = "C:/path/to/LibriSpeech"  # Update this path!
# Folder structure expected:
# LibriSpeech/
#   ├── train-clean-100/
#   ├── dev-clean/
#   └── test-clean/

use_streaming: bool = False  # Not used when use_local_dataset=True
streaming_train_samples: int = None  # Not used when use_local_dataset=True
streaming_eval_samples: int = None
streaming_test_samples: int = None

# Model
base_model: str = "facebook/wav2vec2-base-960h"
vocab_path: str = "vocab.json"

# Custom pronunciation layers
use_custom_head: bool = True
hidden_dropout: float = 0.1
attention_dropout: float = 0.1
intermediate_size: int = 512
num_pronunciation_layers: int = 2  # Additional layers on top of pretrained model
pronunciation_head_dim: int = 256  # Dimension for pronunciation scoring head

# Phoneme configuration
use_ipa_phonemes: bool = True  # Use IPA instead of simple G2P
include_stress_markers: bool = True  # Include stress patterns in phonemes
include_syllable_boundaries: bool = True  # Mark syllable boundaries

# Training
output_dir: str = "./wav2vec2-pronunciation-ctc"
batch_size: int = 4
gradient_accumulation_steps: int = 4
num_epochs: int = 10
learning_rate: float = 1e-4
fp16: bool = True
logging_steps: int = 25
save_total_limit: int = 2
eval_steps: int = 500
save_steps: int = 500
warmup_steps: int = 500  # Warmup for better training
