from transformers import TrainingArguments, Trainer

from config import Config
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
    config = Config()
    device = get_device()
    
    # Load datasets
    datasets = load_librispeech_datasets(config)
    
    # Convert text to phonemes
    print("Converting text to phonemes...")
    datasets = datasets.map(text_to_phoneme_chars)
    
    # Build vocabulary from train dataset
    print("Building vocabulary...")
    vocab_dict = build_vocab(datasets["train"], config.vocab_path)
    
    # Create processor
    processor = create_processor(config.vocab_path, config.sampling_rate)
    
    # Process datasets
    processed_datasets = process_datasets(datasets, processor)
    
    # Create model
    model = create_model(config, processor, device)
    
    # Create data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        report_to="none"
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
    print("Starting training...")
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
