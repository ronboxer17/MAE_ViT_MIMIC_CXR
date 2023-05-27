import PIL
import requests
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torchvision import transforms
from transformers import AutoFeatureExtractor, ViTMAEForPreTraining


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


def build_transform(is_train=True):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=224,
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
    # if args.input_size <= 224:
    #     crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    crop_pct = 1.0
    size = int(224 / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# s = create_train_val_test('train')

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
