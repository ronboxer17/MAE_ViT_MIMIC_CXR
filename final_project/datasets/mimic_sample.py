import os.path
from typing import Any, Optional

from torchvision import datasets

from final_project.config import MIMIC_SAMPLE_ROOT
from final_project.transformers.proj_transformers import DEF_TRANSFORMER


def build_mimic_sample(
    transformer: Optional[Any] = None, is_train: bool = True
) -> datasets.ImageFolder:
    """
    Load the MIMIC sample dataset.
    :param transformer: Optional, a valid transformer to apply to the images.
    :param is_train: A boolean indicating whether to load the train or val subset.
    :return: An instance of ImageFolder dataset.
    """
    subset = "train" if is_train else "val"
    path = os.path.join(MIMIC_SAMPLE_ROOT, subset)

    return datasets.ImageFolder(root=path, transform=transformer or DEF_TRANSFORMER)
