import requests
from PIL import Image
from torch import nn
from transformers import (
    AutoFeatureExtractor,
    ViTMAEForPreTraining,
)


class MAE(nn.Module):
    model_name = "facebook/vit-mae-base"

    def __init__(self, possible_labels: list[str], *args, **kwargs):
        self.possible_labels = possible_labels
        super().__init__(*args, **kwargs)
        self.mae = ViTMAEForPreTraining.from_pretrained(self.model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.mae.config.hidden_size, len(possible_labels)),
        )

    def forward(self, inputs):
        outputs = self.mae.forward(inputs)
        return self.classifier(
            outputs.logits[:, 0, :]  # reshape to (batch_size, hidden_size)
        )


if __name__ == "__main__":
    mae = MAE(possible_labels=["cat", "dog"])
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

    inputs = feature_extractor(images=[image, image], return_tensors="pt")
    outputs = model(**inputs)
    loss = outputs.loss
    mask = outputs.mask
    ids_restore = outputs.ids_restore
