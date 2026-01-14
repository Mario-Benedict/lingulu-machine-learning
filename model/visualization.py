import matplotlib.pyplot as plt


def plot_training_history(trainer, save_path="training_loss.png"):
    """Plot and save training loss history."""
    history = trainer.state.log_history
    
    train_losses = [x["loss"] for x in history if "loss" in x]
    train_epochs = [x["epoch"] for x in history if "loss" in x]
    
    eval_losses = [x["eval_loss"] for x in history if "eval_loss" in x]
    eval_epochs = [x["epoch"] for x in history if "eval_loss" in x]
    
    plt.figure(figsize=(10, 6))
    
    if train_losses:
        plt.plot(train_epochs, train_losses, label="Train Loss", marker='o')
    
    if eval_losses:
        plt.plot(eval_epochs, eval_losses, label="Eval Loss", marker='s')
    
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_per_comparison(train_per, eval_per, test_per, save_path="per_comparison.png"):
    """Plot PER comparison across datasets."""
    datasets = ["Train", "Eval", "Test"]
    per_values = [train_per, eval_per, test_per]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(datasets, per_values, color=['blue', 'orange', 'green'])
    
    for bar, val in zip(bars, per_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{val:.4f}', ha='center', va='bottom')
    
    plt.xlabel("Dataset")
    plt.ylabel("Phoneme Error Rate (PER)")
    plt.title("PER Comparison Across Datasets")
    plt.ylim(0, max(per_values) * 1.2)
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()
