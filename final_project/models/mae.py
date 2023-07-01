from typing import List

import torch
from torch import nn
from transformers import ViTMAEForPreTraining


class MAE(nn.Module):
    model_name = "facebook/vit-mae-base"

    def __init__(
        self,
        possible_labels: List[str] = ["0", "1"],
        fine_tune_only_classifier=False,
        *args,
        **kwargs
    ):
        self.possible_labels = possible_labels

        super().__init__(*args, **kwargs)

        self.mae = ViTMAEForPreTraining.from_pretrained(self.model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.mae.config.hidden_size, len(possible_labels)),
        )
        if fine_tune_only_classifier:
            self._fine_tune_classifier()

    def _fine_tune_classifier(self):
        # this will make only the last encoding layers to be learned
        # set the other layers to be frozen
        for name, param in self.mae.named_parameters():
            param.requires_grad = False

    def forward(self, inputs, device="cpu") -> torch.Tensor:
        # shape = inputs.data.get('pixel_values').shape
        # if len(shape) == 5:
        #     b, _, p, s1, s2 = shape
        #     inputs.data['pixel_values'] = inputs.data['pixel_values'].reshape(b, p, s1, s2)

        outputs = self.mae.forward(inputs.to(device))
        return self.classifier(
            outputs.logits[:, 0, :]  # reshape to (batch_size, hidden_size)
        )
