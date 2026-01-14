import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received."""
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels_batch = self.processor.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch
