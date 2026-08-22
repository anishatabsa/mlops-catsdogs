"""
Torch Dataset / transform definitions for the processed Cats vs Dogs data.

Train split gets data augmentation (random flip, rotation, color jitter);
val/test splits only get the deterministic resize + normalize pipeline,
so evaluation metrics are stable and comparable run over run.
"""
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_folder(root: str, train: bool) -> ImageFolder:
    tfm = train_transforms() if train else eval_transforms()
    return ImageFolder(root=root, transform=tfm)
