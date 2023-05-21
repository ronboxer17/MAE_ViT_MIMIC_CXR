from transformers import AutoFeatureExtractor, ResNetForImageClassification, ResNetModel
import torch
from final_project.models.base import BaseModel
from torch import nn


class ResNet(BaseModel):
    model_name = "microsoft/resnet-18"

    def __init__(self, labels: list[str], device='cpu'):
        self.resnet = ResNetModel.from_pretrained(self.model_name)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.resnet.config.hidden_sizes[-1], len(labels))
        )
        print(self.resnet)
        super().__init__(labels, device)

    def forward(self, *args, **kwargs):
        outputs = self.resnet.forward(*args, **kwargs)
        return self.classifier(outputs.pooler_output)


r = ResNet(['1', '2'])
print(r)

#
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
# print(model.config)
#
# from transformers import AutoImageProcessor, ResNetForImageClassification
# import torch
# from datasets import load_dataset
#
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
#
# image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
#
# inputs = image_processor(image, return_tensors="pt")
#
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])
