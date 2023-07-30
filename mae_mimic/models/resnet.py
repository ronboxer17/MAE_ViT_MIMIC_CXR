from torch import nn
from transformers import ResNetModel

from .datamodels import Models


class ResNet(nn.Module):
    model_name = Models.RESNET18.value

    def __init__(
        self,
        possible_labels: list[str] = ["0", "1"],
        fine_tune_only_classifier=False,
        *args,
        **kwargs
    ):
        self.possible_labels = possible_labels

        super().__init__(*args, **kwargs)

        self.resnet = ResNetModel.from_pretrained(self.model_name)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.resnet.config.hidden_sizes[-1], len(possible_labels)),
        )

        if fine_tune_only_classifier:
            self._fine_tune_classifier()

    def _fine_tune_classifier(self):
        # this will make only the last encoding layers to be learned
        # set the other layers to be frozen
        for name, param in self.resnet.named_parameters():
            param.requires_grad = False

    def forward(self, inputs, device="cpu"):
        outputs = self.resnet.forward(inputs.to(device))
        return self.classifier(outputs.pooler_output)
