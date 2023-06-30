import os.path
from typing import Any, Optional

from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

from final_project.config import DATA_PATH, MIMIC_SAMPLE_ROOT

DEF_TRANSFORMER = transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_mimic_sample(transformer: Optional[Any] = None, is_train: bool = True) -> datasets.ImageFolder:
    """
    Load the MIMIC sample dataset.

    :param transformer: Optional, a valid transformer to apply to the images.
    :param is_train: A boolean indicating whether to load the train or val subset.
    :return: An instance of ImageFolder dataset.
    """
    subset = "train" if is_train else "val"
    path = os.path.join(MIMIC_SAMPLE_ROOT, subset)

    print(path)
    return datasets.ImageFolder(
        root=path,
        transform=transformer or DEF_TRANSFORMER
    )

#
# train = build_mimic_sample()
#
# for batch in DataLoader(train, batch_size=3):
#     print(batch)
# #     break
# from PIL import Image
#
# image = Image.open(r'D:\ron\mimic\final_project\final_project\assets\data\mimic-cxr\mimic_sample\train\0\0a7cb25c-aa0ea467-5842af02-edf6d794-068659d4.jpg').convert("RGB")
# print(image)