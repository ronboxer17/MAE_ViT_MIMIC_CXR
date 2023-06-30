from final_project.datasets.mimic import build_mimic_dataset
from torch.utils.data import DataLoader, Dataset
from pydantic import BaseModel
from final_project.config import (
    IDS_TO_IMAGES_PATHS,
)
import json

with open(IDS_TO_IMAGES_PATHS, "r") as f:
    ids_to_paths = json.load(f)


class ImageMetadata(BaseModel):
    id: str
    label: int
    path: str


transformer_model = "facebook/vit-mae-base"
train_dataset = build_mimic_dataset(transformer_model, is_train=True)
val_dataset = build_mimic_dataset(transformer_model, is_train=False)




sample_size = 2000

sample_images = []

for i, batch in enumerate(DataLoader(val_dataset, batch_size=1, shuffle=True)):
    id = batch[2][0]
    path = ids_to_paths.get(id)
    label = int(batch[1][0])
    img = ImageMetadata(id=id, label=label, path=path)
    sample_images.append(img.dict())
    print(i)
    if i == sample_size:
        with open(f'val_sample_{sample_size}.json', 'w') as f:
            json.dump(json.dumps(sample_images), f)
        break

    with open(f'val_sample_{sample_size}.json', 'w') as f:
        json.dump(json.dumps(sample_images), f)