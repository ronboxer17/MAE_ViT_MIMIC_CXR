import os.path
from typing import Any, Optional

from torchvision import datasets
from torch.utils.data import Subset
import torch
from final_project.config import MIMIC_SAMPLE_ROOT, SEED
from final_project.transformers.proj_transformers import DEF_TRANSFORMER


def build_mimic_sample(
    transformer: Optional[Any] = None,
    sample_size: int = 10000,
    is_train: bool = True
) -> Subset:
    """
    Load the MIMIC sample dataset.
    :param sample_size:
    :param transformer: Optional, a valid transformer to apply to the images.
    :param is_train: A boolean indicating whether to load the train or val subset.
    :return: An instance of ImageFolder dataset.
    """
    dataset_type = "train" if is_train else "val"
    dataset = datasets.ImageFolder(
        root=os.path.join(MIMIC_SAMPLE_ROOT, dataset_type),
        transform=transformer or DEF_TRANSFORMER
    )

    assert len(dataset) >= sample_size, f"Sample size {sample_size} is larger than dataset size {len(dataset)}"

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator)

    return Subset(dataset, indices[:sample_size].tolist())

