import os.path
from typing import Tuple

from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

from final_project.config import DATA_PATH

DATA_DIR = "parking/data"
data_root_path = os.path.join(DATA_PATH, DATA_DIR)
transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_parking_dataset(
        root: str = data_root_path, transform=transformer
) -> datasets.ImageFolder:
    return datasets.ImageFolder(root=root, transform=transform)


def train_test_mock_data(
        train_ratio=0.8,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    dataset = load_parking_dataset()
    train_size = int(train_ratio * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    from final_project.utils import display_image

    data, _ = train_test_mock_data()
    dataloader = DataLoader(data, batch_size=1)
    for batch in dataloader:
        print(batch)
        print()
        break
