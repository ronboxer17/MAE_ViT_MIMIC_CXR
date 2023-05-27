import os.path

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
