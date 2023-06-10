from torch import nn
from transformers import ResNetModel


class ResNet(nn.Module):
    model_name = "microsoft/resnet-18"

    def __init__(self, possible_labels: list[str] = ["0", "1"], *args, **kwargs):
        self.possible_labels = possible_labels

        super().__init__(*args, **kwargs)

        self.resnet = ResNetModel.from_pretrained(self.model_name)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.resnet.config.hidden_sizes[-1], len(possible_labels)),
        )

    def forward(self, inputs, *args):
        outputs = self.resnet.forward(inputs)
        return self.classifier(outputs.pooler_output)
