import re
import json
import nltk
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from g2p_en import G2p
import eng_to_ipa as ipa  # For IPA conversion
import librosa
import soundfile as sf
import config

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
    """Load audio using librosa when audio column has bytes or path."""
    import io
    
    # Handle both streaming (bytes) and local (path) formats
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
    elif isinstance(batch["audio"], str):
        # Direct path string (local dataset)
        audio_array, _ = librosa.load(
            batch["audio"],
            sr=config.sampling_rate,
            mono=True
        )
    else:
        raise ValueError("Unexpected audio format")
    
    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = librosa.to_mono(audio_array.T)
    
    batch["speech"] = audio_array
    batch["sampling_rate"] = config.sampling_rate
    return batch


def load_local_librispeech_dataset(dataset_path: str, split_name: str):
    """
    Load LibriSpeech dataset from local folder structure.
    
    Expected structure:
    LibriSpeech/
      train-clean-100/
        speaker_id/
          chapter_id/
            speaker_id-chapter_id-utterance_id.flac
            speaker_id-chapter_id.trans.txt
      dev-clean/
      test-clean/
    
    Args:
        dataset_path: Path to LibriSpeech root folder
        split_name: 'train-clean-100', 'dev-clean', or 'test-clean'
    
    Returns:
        Dataset object with audio paths and transcripts
    """
    split_path = Path(dataset_path) / split_name
    
    if not split_path.exists():
        raise ValueError(f"Dataset split not found: {split_path}")
    
    print(f"Loading {split_name} from {split_path}...")
    
    data = {
        "audio": [],
        "text": [],
        "speaker_id": [],
        "chapter_id": [],
        "utterance_id": []
    }
    
    # Walk through speaker directories
    speaker_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    
    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        
        # Walk through chapter directories
        chapter_dirs = sorted([d for d in speaker_dir.iterdir() if d.is_dir()])
        
        for chapter_dir in chapter_dirs:
            chapter_id = chapter_dir.name
            
            # Read transcript file
            trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            
            if not trans_file.exists():
                print(f"Warning: Transcript file not found: {trans_file}")
                continue
            
            # Parse transcripts
            transcripts = {}
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            utt_id, text = parts
                            transcripts[utt_id] = text
            
            # Find all audio files
            audio_files = sorted(chapter_dir.glob(f"{speaker_id}-{chapter_id}-*.flac"))
            
            for audio_file in audio_files:
                # Extract utterance ID from filename
                utt_id = audio_file.stem  # e.g., "61-70968-0000"
                
                if utt_id in transcripts:
                    data["audio"].append(str(audio_file))
                    data["text"].append(transcripts[utt_id])
                    data["speaker_id"].append(speaker_id)
                    data["chapter_id"].append(chapter_id)
                    data["utterance_id"].append(utt_id)
                else:
                    print(f"Warning: No transcript found for {audio_file}")
    
    print(f"✓ Loaded {len(data['audio'])} samples from {split_name}")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(data)
    
    return dataset

def load_librispeech_datasets():
    """Load train, validation, and test datasets from LibriSpeech."""
    use_local = getattr(config, 'use_local_dataset', False)
    
    if use_local:
        # Load from local folder
        local_path = getattr(config, 'local_dataset_path', None)
        
        if not local_path:
            raise ValueError(
                "local_dataset_path not set in config.py! "
                "Please set: local_dataset_path = 'C:/path/to/LibriSpeech'"
            )
        
        print("=" * 60)
        print("📁 Loading LOCAL LibriSpeech dataset")
        print(f"📂 Path: {local_path}")
        print("=" * 60)
        
        # Load train dataset
        train_dataset = load_local_librispeech_dataset(local_path, "train-clean-100")
        
        # Load validation dataset
        eval_dataset = load_local_librispeech_dataset(local_path, "dev-clean")
        
        # Load test dataset
        test_dataset = load_local_librispeech_dataset(local_path, "test-clean")
        
        print("\n✅ All datasets loaded from local storage")
        print(f"  ├─ Train samples: {len(train_dataset)}")
        print(f"  ├─ Eval samples: {len(eval_dataset)}")
        print(f"  └─ Test samples: {len(test_dataset)}")
        print()
        
        return {
            "train": train_dataset,
            "eval": eval_dataset,
            "test": test_dataset
        }
    
    else:
        # Original streaming/download mode
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
    """Process all datasets with the processor (local and streaming compatible)."""
    use_local = getattr(config, 'use_local_dataset', False)
    prepare_fn = prepare_dataset_fn(processor)
    processed = {}
    
    for split_name, dataset in datasets.items():
        print(f"\nProcessing {split_name} dataset...")
        
        if use_local:
            # Local dataset: direct processing
            print(f"  Step 1/2: Loading audio from local files...")
            dataset_with_audio = dataset.map(
                load_audio_librosa,
                desc=f"Loading {split_name} audio"
            )
            
            print(f"  Step 2/2: Extracting features and tokenizing...")
            processed_dataset = dataset_with_audio.map(
                prepare_fn,
                remove_columns=["speech", "sampling_rate"],
                desc=f"Processing {split_name}"
            )
        else:
            # Streaming dataset: process as before
            print(f"  Step 1/2: Loading audio from stream...")
            dataset_with_audio = dataset.map(load_audio_librosa)
            
            print(f"  Step 2/2: Extracting features and tokenizing...")
            processed_dataset = dataset_with_audio.map(
                prepare_fn,
                remove_columns=["speech", "sampling_rate"],
            )
        
        processed[split_name] = processed_dataset
        print(f"✓ {split_name} dataset ready")
    
    return processed
