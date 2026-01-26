import re
import json
import nltk
from datasets import load_dataset, DatasetDict, disable_caching
from g2p_en import G2p
import eng_to_ipa as ipa  # For IPA conversion
import librosa
import config

disable_caching()

nltk.download('averaged_perceptron_tagger_eng', quiet=True)

g2p = G2p()


def text_to_ipa_phonemes(text: str, include_stress: bool = True, include_syllables: bool = True):
    """
    Convert text to IPA (International Phonetic Alphabet) phonemes.
    More accurate for pronunciation assessment than simple G2P.
    
    Args:
        text: Input text to convert
        include_stress: Include stress markers (ˈ for primary, ˌ for secondary)
        include_syllables: Include syllable boundary markers (.)
    
    Returns:
        String of IPA phonemes
    """
    # Clean text
    text = re.sub(r"[^a-zA-Z ]", "", text).lower()
    
    try:
        # Convert to IPA using eng_to_ipa library
        ipa_text = ipa.convert(text)
        
        # If conversion failed, fall back to G2P
        if ipa_text == text or ipa_text == "*":
            phonemes = g2p(text)
            phonemes = [p.lower() for p in phonemes if p.isalpha()]
            return "".join(phonemes)
        
        # Process IPA text
        result = ipa_text
        
        if not include_stress:
            # Remove stress markers
            result = result.replace("ˈ", "").replace("ˌ", "")
        
        if not include_syllables:
            # Remove syllable boundaries
            result = result.replace(".", "")
        
        # Remove spaces and keep only phonetic characters
        result = result.replace(" ", "|")  # Use | as word boundary
        
        return result
        
    except Exception as e:
        # Fallback to G2P if IPA conversion fails
        print(f"IPA conversion failed for '{text}': {e}. Using G2P fallback.")
        phonemes = g2p(text)
        phonemes = [p.lower() for p in phonemes if p.isalpha()]
        return "".join(phonemes)


def text_to_phoneme_chars(batch):
    """
    Convert text to phoneme characters.
    Uses IPA for pronunciation if configured, otherwise falls back to G2P.
    """
    config_module = config
    
    if hasattr(config_module, 'use_ipa_phonemes') and config_module.use_ipa_phonemes:
        # Use IPA phonemes for pronunciation assessment
        include_stress = getattr(config_module, 'include_stress_markers', True)
        include_syllables = getattr(config_module, 'include_syllable_boundaries', True)
        
        batch["phonemes"] = text_to_ipa_phonemes(
            batch["text"],
            include_stress=include_stress,
            include_syllables=include_syllables
        )
    else:
        # Original G2P method
        text = re.sub(r"[^a-zA-Z ]", "", batch["text"]).lower()
        phonemes = g2p(text)
        phonemes = [p.lower() for p in phonemes if p.isalpha()]
        batch["phonemes"] = "".join(phonemes)
    
    return batch


def extract_vocab(batch):
    """Extract vocabulary from phonemes."""
    all_text = "".join(batch["phonemes"])
    vocab = list(set(all_text))
    return {"vocab": vocab}


def build_vocab(dataset, vocab_path: str):
    """Build and save vocabulary from dataset (streaming compatible)."""
    print("Building vocabulary from phonemes...")
    
    # Collect all unique phoneme characters from streaming dataset
    vocab_chars = set()
    sample_count = 0
    
    for batch in dataset:
        if "phonemes" in batch:
            vocab_chars.update(list(batch["phonemes"]))
            sample_count += 1
            if sample_count % 1000 == 0:
                print(f"  Processed {sample_count} samples, {len(vocab_chars)} unique chars")
    
    vocab_list = sorted(list(vocab_chars))
    vocab_dict = {v: i for i, v in enumerate(vocab_list)}
    
    # Add special tokens
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)
    
    print(f"✓ Vocab size: {len(vocab_dict)} (from {sample_count} samples)")
    return vocab_dict

def load_audio_librosa(batch):
    """Load audio using librosa when audio column has bytes."""
    import io
    import soundfile as sf
    
    # Handle both streaming (bytes) and downloaded (path) formats
    if isinstance(batch["audio"], dict):
        if "bytes" in batch["audio"] and batch["audio"]["bytes"] is not None:
            # Streaming mode - decode from bytes
            audio_bytes = batch["audio"]["bytes"]
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Resample if needed
            if sr != config.sampling_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sr,
                    target_sr=config.sampling_rate
                )
        elif "path" in batch["audio"] and batch["audio"]["path"] is not None:
            # Downloaded mode - load from path
            audio_array, _ = librosa.load(
                batch["audio"]["path"],
                sr=config.sampling_rate,
                mono=True
            )
        else:
            raise ValueError("Audio data not found in expected format")
    else:
        raise ValueError("Unexpected audio format")
    
    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = librosa.to_mono(audio_array.T)
    
    batch["speech"] = audio_array
    batch["sampling_rate"] = config.sampling_rate
    return batch

def load_librispeech_datasets():
    """Load train, validation, and test datasets from LibriSpeech using streaming."""
    use_streaming = getattr(config, 'use_streaming', True)
    
    if use_streaming:
        print("🚀 Loading datasets in STREAMING mode (no download needed)...")
        print("💾 Storage used: 0GB")
        print("⚠️  Internet required during training\n")
    else:
        print("📥 Loading datasets in DOWNLOAD mode...")
        print("⚠️  This will download ~6.5GB of data\n")
    
    # Load train dataset
    print("Loading train split...")
    train_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
        streaming=use_streaming,
    )
    
    # Load validation dataset
    print("Loading validation split...")
    eval_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        streaming=use_streaming,
    )
    
    # Load test dataset
    print("Loading test split...")
    test_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        streaming=use_streaming,
    )
    
    # Apply sample limits if streaming
    if use_streaming:
        train_limit = getattr(config, 'streaming_train_samples', None)
        eval_limit = getattr(config, 'streaming_eval_samples', None)
        test_limit = getattr(config, 'streaming_test_samples', None)
        
        if train_limit:
            train_dataset = train_dataset.take(train_limit)
            print(f"  ├─ Train: limited to {train_limit} samples")
        else:
            print(f"  ├─ Train: using ALL samples")
            
        if eval_limit:
            eval_dataset = eval_dataset.take(eval_limit)
            print(f"  ├─ Eval: limited to {eval_limit} samples")
        else:
            print(f"  ├─ Eval: using ALL samples")
            
        if test_limit:
            test_dataset = test_dataset.take(test_limit)
            print(f"  └─ Test: limited to {test_limit} samples")
        else:
            print(f"  └─ Test: using ALL samples")
        
        print("\n✅ Datasets loaded in streaming mode (using full dataset)")
        print("💡 Tip: Edit config.py to set limits if needed")
    else:
        print(f"\n✅ Datasets downloaded")
        print(f"  ├─ Train samples: {len(train_dataset)}")
        print(f"  ├─ Eval samples: {len(eval_dataset)}")
        print(f"  └─ Test samples: {len(test_dataset)}")
    
    print()
    
    return {
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset
    }


def prepare_dataset_fn(processor):
    """Return a function to prepare dataset samples (streaming compatible)."""
    def prepare_dataset(batch):
        # Use pre-loaded speech from load_audio_librosa
        if "speech" in batch:
            audio_array = batch["speech"]
        else:
            raise ValueError("Audio not loaded. Run load_audio_librosa first.")
        
        # Process audio with feature extractor
        batch["input_values"] = processor(
            audio_array,
            sampling_rate=config.sampling_rate
        ).input_values[0]
        
        # Encode phonemes to labels
        batch["labels"] = processor.tokenizer(batch["phonemes"]).input_ids
        return batch
    
    return prepare_dataset


def process_datasets(datasets, processor):
    """Process all datasets with the processor (streaming compatible)."""
    from datasets import Dataset
    from itertools import islice
    
    prepare_fn = prepare_dataset_fn(processor)
    processed = {}
    
    for split_name, dataset in datasets.items():
        print(f"\nProcessing {split_name} dataset...")
        
        # Load audio first (decode bytes to numpy arrays)
        print(f"  Step 1/2: Loading audio from stream...")
        dataset_with_audio = dataset.map(load_audio_librosa)
        
        # Then prepare for model (extract features, tokenize phonemes)
        print(f"  Step 2/2: Extracting features and tokenizing...")
        processed_dataset = dataset_with_audio.map(
            prepare_fn,
            remove_columns=["speech", "sampling_rate"],  # Keep only input_values and labels
        )
        
        processed[split_name] = processed_dataset
        print(f"✓ {split_name} dataset ready")
    
    return processed
