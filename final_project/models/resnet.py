from torch import nn
from transformers import ResNetModel


class ResNet(nn.Module):
    model_name = "microsoft/resnet-18"

    def __init__(self, possible_labels: list[str] = ["0", "1"], fine_tune_only_classifier=True, *args, **kwargs):
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

    def forward(self, inputs, device='cpu'):
        shape = inputs.data.get('pixel_values').shape
        if len(shape) == 5:
            b, _, p, s1, s2 = shape
            inputs.data['pixel_values'] = inputs.data['pixel_values'].reshape(b, p, s1, s2)

        outputs = self.resnet.forward(
            inputs.data.get('pixel_values').to(device)
        )
        return self.classifier(outputs.pooler_output)
