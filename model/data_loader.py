import re
import json
import nltk
from datasets import load_dataset, Audio, DatasetDict
from g2p_en import G2p

nltk.download('averaged_perceptron_tagger_eng', quiet=True)

g2p = G2p()


def text_to_phoneme_chars(batch):
    """Convert text to phoneme characters."""
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


def load_librispeech_datasets(config):
    """Load train, validation, and test datasets from LibriSpeech."""
    print("Loading datasets...")
    
    # Load train dataset
    train_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
        streaming=False
    )
    
    # Load validation dataset (dev-clean)
    eval_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        streaming=False
    )
    
    # Load test dataset (test-clean)
    test_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        streaming=False
    )

    # Cast audio column to correct sampling rate
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
    
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
