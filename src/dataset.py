import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_metadata() -> pd.DataFrame:
    """Read the metadata file and add a numeric label and image path to each row."""
    df = pd.read_csv(config.METADATA_PATH, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    df["label"] = df["dx"].map(class_to_idx)
    df["path"] = df["image_id"].apply(lambda x: config.IMAGE_DIR / f"{x}.jpg")
    assert df["label"].notna().all(), "Found a diagnosis not listed in CLASS_NAMES"
    return df


def split_by_lesion(df: pd.DataFrame):
    """Split into train/val/test so that all images of one lesion stay together."""
    lesions = df.groupby("lesion_id")["dx"].first().reset_index()
    holdout = config.VAL_FRACTION + config.TEST_FRACTION

    train_les, temp_les = train_test_split(
        lesions, test_size=holdout, stratify=lesions["dx"], random_state=config.SEED
    )
    test_share = config.TEST_FRACTION / holdout
    val_les, test_les = train_test_split(
        temp_les, test_size=test_share, stratify=temp_les["dx"], random_state=config.SEED
    )

    train_df = df[df["lesion_id"].isin(train_les["lesion_id"])]
    val_df = df[df["lesion_id"].isin(val_les["lesion_id"])]
    test_df = df[df["lesion_id"].isin(test_les["lesion_id"])]
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_transforms(train: bool):
    """Augmentation for training; plain resize and normalize for val/test."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def get_dataloaders():
    df = load_metadata()
    train_df, val_df, test_df = split_by_lesion(df)

    train_ds = HAM10000Dataset(train_df, get_transforms(train=True))
    val_ds = HAM10000Dataset(val_df, get_transforms(train=False))
    test_ds = HAM10000Dataset(test_df, get_transforms(train=False))

    kwargs = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader