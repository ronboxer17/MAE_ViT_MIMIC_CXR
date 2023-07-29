from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from transformers.proj_transformers import DEF_TRANSFORMER

from mae_mimic.config import PARKING_DATA_PATH


def load_parking_dataset(transformer=None) -> datasets.ImageFolder:
    return datasets.ImageFolder(
        root=PARKING_DATA_PATH, transform=transformer or DEF_TRANSFORMER
    )


def train_test_mock_data(
    train_ratio=0.8, transformer=None
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    dataset = load_parking_dataset(transformer=transformer)
    train_size = int(train_ratio * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    data, _ = train_test_mock_data()
    dataloader = DataLoader(data, batch_size=1)
    for batch in dataloader:
        print(batch)
        print()
        break
