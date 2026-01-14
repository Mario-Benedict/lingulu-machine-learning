import torch
from jiwer import cer


def compute_metrics_fn(processor):
    """Return a function to compute metrics during evaluation."""
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
        
        # Replace -100 with pad token id
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
        phoneme_error_rate = cer(label_str, pred_str)
        
        return {"per": phoneme_error_rate}
    
    return compute_metrics


def evaluate_model(model, dataset, processor, device, n_samples=300):
    """Evaluate model on dataset and compute PER."""
    model.eval()
    preds, refs = [], []
    
    n_samples = min(n_samples, len(dataset))
    
    for i in range(n_samples):
        audio = dataset[i]["input_values"]
        labels = dataset[i]["labels"]

        with torch.no_grad():
            logits = model(
                torch.tensor(audio).unsqueeze(0).to(device)
            ).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(pred_ids)[0]

        ref = processor.decode(
            [x for x in labels if x != -100]
        )

        preds.append(pred)
        refs.append(ref)

    per = cer(refs, preds)
    print(f"Phoneme Error Rate (PER): {per:.4f}")
    
    return preds, refs, per
