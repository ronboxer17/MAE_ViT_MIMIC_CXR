from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from final_project.config import PARKING_DATA_PATH
from final_project.proj_transformers import DEF_TRANSFORMER


def load_parking_dataset(transformer=None) -> datasets.ImageFolder:
    return datasets.ImageFolder(
        root=PARKING_DATA_PATH,
        transform=transformer or DEF_TRANSFORMER
    )


def train_test_mock_data(train_ratio=0.8, transformer=None) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    dataset = load_parking_dataset(transformer=transformer)
    train_size = int(train_ratio * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    from final_project.utils.plots import display_image

    data, _ = train_test_mock_data()
    dataloader = DataLoader(data, batch_size=1)
    for batch in dataloader:
        print(batch)
        print()
        break
