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
    """Build and save vocabulary from dataset."""
    vocabs = dataset.map(
        extract_vocab,
        batched=True,
        batch_size=-1,
        remove_columns=dataset.column_names
    )
    
    vocab_list = sorted(list(set(vocabs["vocab"])))
    vocab_dict = {v: i for i, v in enumerate(vocab_list)}
    
    # Add special tokens
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)
    
    print(f"Vocab size: {len(vocab_dict)}")
    return vocab_dict

def load_audio_librosa(batch):
    path = batch["audio"]["path"]

    speech, _ = librosa.load(
        path,
        sr=config.sampling_rate,
        mono=True
    )

    batch["speech"] = speech
    batch["sampling_rate"] = config.sampling_rate
    return batch

def load_librispeech_datasets():
    """Load train, validation, and test datasets from LibriSpeech."""
    print("Loading datasets...")
    
    # Load train dataset
    train_dataset = load_dataset(
        "librispeech_asr",
        "train_clean_100",
        split="train",
    )
    
    # Load validation dataset (dev-clean)
    eval_dataset = load_dataset(
        "librispeech_asr",
        "dev_clean",
        split="validation",
    )
    
    # Load test dataset (test-clean)
    test_dataset = load_dataset(
        "librispeech_asr",
        "test_clean",
        split="test",
    )

    train_dataset = train_dataset.map(
        load_audio_librosa,
        remove_columns=train_dataset.column_names,
        num_proc=1
    )

    eval_dataset = eval_dataset.map(
        load_audio_librosa,
        remove_columns=eval_dataset.column_names,
        num_proc=1
    )

    test_dataset = test_dataset.map(
        load_audio_librosa,
        remove_columns=test_dataset.column_names,
        num_proc=1
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset
    })


def prepare_dataset_fn(processor):
    """Return a function to prepare dataset samples."""
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"],
            sampling_rate=16000
        ).input_values[0]
        
        batch["labels"] = processor.tokenizer(batch["phonemes"]).input_ids
        return batch
    
    return prepare_dataset


def process_datasets(datasets, processor):
    """Process all datasets with the processor."""
    prepare_fn = prepare_dataset_fn(processor)
    
    processed = DatasetDict()
    for split_name, dataset in datasets.items():
        print(f"Processing {split_name} dataset...")
        processed[split_name] = dataset.map(
            prepare_fn,
            remove_columns=dataset.column_names,
            num_proc=1
        )
    
    return processed
