import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from final_project.config import (IDS_TO_IMAGES_PATHS,
                                  IDS_WITH_LABELS_AND_SPLITS, MIMIC_FILES_PATH)
from final_project.datasets.datamodels import MimicImage
from final_project.models.mae import build_transform

with open(IDS_TO_IMAGES_PATHS, "r") as f:
    IDS_TO_IMAGES = json.load(f)


class MimicDataset(Dataset):
    def __init__(self, ids_labels: List[MimicImage], transform=None):
        with open(IDS_TO_IMAGES_PATHS, "r") as f:
            self.ids_to_images = json.load(f)
        self.ids_labels = ids_labels
        self.transform = transform

    def __len__(self):
        return len(self.ids_labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        mimic_image = self.ids_labels[idx]
        id = mimic_image.id
        label = mimic_image.label
        path = self.ids_to_images.get(id)
        image = Image.open(os.path.join(MIMIC_FILES_PATH, path)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_train_val_test(split: str, view: str = "frontal") -> List[MimicImage]:
    assert split in ["train", "val", "test"]
    df = pd.read_csv(IDS_WITH_LABELS_AND_SPLITS)
    df = df[df.split == split]
    df = df[df.view == view]
    df = df[~df.findings.isna()]
    df = df[["dicom_id", "view", "findings"]]
    return df.apply(lambda x: MimicImage(x.dicom_id, int(x.findings)), axis=1).tolist()


def load_mimic_dataset(is_train=True):
    transform = build_transform(is_train)
    # TODO: check 'val' case
    train_ids = create_train_val_test("train" if is_train else "val")
    dataset = MimicDataset(train_ids, transform=transform)

    print(dataset)

    return dataset


if __name__ == "__main__":
    data = load_mimic_dataset()
    dataloader = DataLoader(data, batch_size=1)
    for batch in dataloader:
        print(batch)
        print()
