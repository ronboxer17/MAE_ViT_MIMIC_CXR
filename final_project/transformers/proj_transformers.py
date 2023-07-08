from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import PIL

INPUT_SIZE = 224

DEF_TRANSFORMER = transformer = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

classic_augmentation_transformer = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

new_augmentation_transformer = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ], p=0.5),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_mimic_transform(is_train=True, input_size=INPUT_SIZE):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=None,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0

    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


AVAILABLE_TRANSFORMS = {
    "default": {
        'train': DEF_TRANSFORMER,
        'val': DEF_TRANSFORMER
    },
    "mimic": {
        'train': build_mimic_transform(is_train=True),
        'val': build_mimic_transform(is_train=False)
    },
    "classic_augmentation": {
        'train': classic_augmentation_transformer,
        'val': DEF_TRANSFORMER
    },
    "new_augmentation": {
        'train': new_augmentation_transformer,
        'val': DEF_TRANSFORMER
    },
}
