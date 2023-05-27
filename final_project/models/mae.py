import requests
from PIL import Image
from torch import nn
from transformers import (
    AutoFeatureExtractor,
    ViTMAEForPreTraining,
)
import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from final_project.datasets.datamodels import MimicImage
# from final_project.datasets.mimic import MimicDataset
import pandas as pd
from final_project.config import MIMIC_FILES_PATH, IDS_TO_IMAGES_PATHS, IDS_WITH_LABELS_AND_SPLITS
from typing import List, Tuple, Optional
import json


# with open(IDS_TO_IMAGES_PATHS, 'r') as f:
#     IDS_TO_IMAGES = json.load(f)
#
#
# def create_mimic_image_if_possible(id: str, label: str) -> Optional[MimicImage]:
#     if IDS_TO_IMAGES.get(id):
#         return MimicImage(id, int(label))
#
#
# def create_train_val_test(split: str, view: str = 'frontal') -> List[MimicImage]:
#     assert split in ['train', 'val', 'test']
#     if split == 'val':  # Special case for MIMIC dataset.
#         split = 'validate'
#     df = pd.read_csv(IDS_WITH_LABELS_AND_SPLITS)
#     df = df[df.split == split]
#     df = df[df.view == view]
#     df = df[~df.findings.isna()]
#     df = df[['dicom_id', 'view', 'findings']]
#     df['mimic_images'] = df.apply(lambda x: create_mimic_image_if_possible(x.dicom_id, x.findings), axis=1)
#     return df[~df.mimic_images.isna()].mimic_images.tolist()
#
#
# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)
#
#     # root = os.path.join(args.data_path, 'train' if is_train else 'val')
#     # dataset = datasets.ImageFolder(root, transform=transform)
#     #TODO- check 'val' case
#     train_ids = create_train_val_test('train' if is_train else 'val')
#     dataset = MimicDataset(train_ids, transform=transform)
#
#     print(dataset)
#
#     return dataset


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
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
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
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
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
