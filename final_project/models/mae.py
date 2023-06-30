from typing import List
import PIL
import requests
import torch
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torchvision import transforms
from transformers import AutoFeatureExtractor, ViTMAEForPreTraining


class MAE(nn.Module):
    # Todo: change to relevant model
    model_name = "facebook/vit-mae-base"

    def __init__(self, possible_labels: List[str] = ["0", "1"], fine_tune_only_classifier=True, *args, **kwargs):
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

    def forward(self, inputs, device='cpu') -> torch.Tensor:
        # shape = inputs.data.get('pixel_values').shape
        # if len(shape) == 5:
        #     b, _, p, s1, s2 = shape
        #     inputs.data['pixel_values'] = inputs.data['pixel_values'].reshape(b, p, s1, s2)
        outputs = self.mae.forward(
            inputs.to(device)
        )
        return self.classifier(
            outputs.logits[:, 0, :]  # reshape to (batch_size, hidden_size)
        )


def build_transform(is_train=True, input_size=224):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=None,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0

    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == "__main__":

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    #
    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
    # model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    #
    # inputs = feature_extractor(images=[image, image], return_tensors="pt")
    # outputs = model(**inputs)
    # loss = outputs.loss
    # mask = outputs.mask
    # ids_restore = outputs.ids_restore

    mae = MAE()
    for name, param in mae.named_parameters():
        print(name, param.requires_grad)
    # print(model)