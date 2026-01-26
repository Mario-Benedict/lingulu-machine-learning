from transformers import TrainingArguments, Trainer

import config
from data_loader import (
    load_librispeech_datasets,
    text_to_phoneme_chars,
    build_vocab,
    process_datasets
)
from model import get_device, create_processor, create_model
from data_collator import DataCollatorCTCWithPadding
from metrics import compute_metrics_fn, evaluate_model
from visualization import plot_training_history, plot_per_comparison


def main():
    # Initialize config and device
    device = get_device()
    
    print("\n=== Pronunciation Assessment Training ===")
    print(f"Using IPA phonemes: {config.use_ipa_phonemes}")
    print(f"Custom pronunciation layers: {config.num_pronunciation_layers}")
    print(f"Stress markers: {config.include_stress_markers}")
    print(f"Syllable boundaries: {config.include_syllable_boundaries}")
    
    # Load datasets
    datasets = load_librispeech_datasets()
    
    # Convert text to phonemes (using IPA if configured)
    print("\nConverting text to phonemes...")
    print("Processing streaming datasets...")
    
    # Map text_to_phoneme_chars to streaming datasets
    datasets_with_phonemes = {}
    for split_name, dataset in datasets.items():
        print(f"  Converting {split_name} to phonemes...")
        datasets_with_phonemes[split_name] = dataset.map(text_to_phoneme_chars)
    
    # Build vocabulary from train dataset (streaming compatible)
    print("\nBuilding vocabulary from train dataset...")
    vocab_dict = build_vocab(datasets_with_phonemes["train"], config.vocab_path)
    
    # Create processor
    processor = create_processor(config.vocab_path, config.sampling_rate)
    
    # Process datasets (load audio, extract features, tokenize)
    print("\nPreparing datasets for training...")
    processed_datasets = process_datasets(datasets_with_phonemes, processor)
    
    # Create custom model with pronunciation layers
    model = create_model(config, processor, device)
    
    # Create data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
    # Training arguments with warmup
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,  # Added warmup for better training
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,  # Save memory for larger model
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["eval"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics_fn(processor)
    )
    
    # Train
    print("\n=== Starting Training ===")
    print("Training custom pronunciation assessment model...")
    trainer.train()
    
    # Save model
    trainer.save_model(f"{config.output_dir}/final")
    processor.save_pretrained(f"{config.output_dir}/final")
    
    # Plot training history
    plot_training_history(trainer)
    
    # Evaluate on all datasets
    print("\n=== Evaluation Results ===")
    
    print("\nEvaluating on train dataset...")
    _, _, train_per = evaluate_model(
        model, processed_datasets["train"], processor, device, n_samples=300
    )
    
    print("\nEvaluating on eval dataset...")
    _, _, eval_per = evaluate_model(
        model, processed_datasets["eval"], processor, device, n_samples=300
    )
    
    print("\nEvaluating on test dataset...")
    preds, refs, test_per = evaluate_model(
        model, processed_datasets["test"], processor, device, n_samples=300
    )
    
    # Plot PER comparison
    plot_per_comparison(train_per, eval_per, test_per)
    
    # Print some examples
    print("\n=== Sample Predictions (Test Set) ===")
    for i in range(min(5, len(preds))):
        print(f"Ref:  {refs[i]}")
        print(f"Pred: {preds[i]}")
        print("-" * 50)

if __name__ == "__main__":
    main()
