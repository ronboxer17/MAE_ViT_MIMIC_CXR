from typing import List

import torch
from torch import nn
from transformers import AutoModel


class ConvNext(nn.Module):
    model_name = "facebook/convnext-base-224"

    def __init__(self, possible_labels: List[str] = ["0", "1"], *args, **kwargs):
        self.possible_labels = possible_labels

        super().__init__(*args, **kwargs)

        self.model = AutoModel.from_pretrained(self.model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_sizes[-1], 768),
            nn.Dropout(0.1),
            nn.Linear(768, len(possible_labels)),
        )

    def forward(self, inputs: torch.Tensor, device="cpu") -> torch.Tensor:
        outputs = self.model.forward(inputs.to(device))
        return self.classifier(outputs.pooler_output)
