import random
from pathlib import Path
from typing import Callable, List, Dict, Union

import pandas as pd
from monai.data import Dataset

from spectre.data._base_datasets import PersistentDataset, GDSDataset


def _initialize_dataset(
    data_dir: str,
    subset: str = "train",
    split_ratio: tuple = (0.8, 0.1, 0.1),  # train, val, test
    seed: int = 0,
) -> List[Dict[str, Union[str, int]]]:
    
    labels_df = pd.read_csv(Path(data_dir) / "labels.csv", index_col="patient_id")
    labels_df["abnormal"] = labels_df["label"].apply(lambda x: int(x.split(",")[1]))
    labels_df = labels_df[["abnormal"]]

    image_paths = sorted(Path(data_dir).glob("**, **, reconstructed_image.nii.gz"))

    random.seed(seed)
    random.shuffle(image_paths)

    train_split_idx = int(split_ratio[0] * len(image_paths))
    val_split_idx = int((split_ratio[0] + split_ratio[1]) * len(image_paths))

    # Split the data into train, val, and test sets
    if subset == "train":
        patient_dirs = patient_dirs[:train_split_idx]
    elif subset == "val":
        patient_dirs = patient_dirs[train_split_idx:val_split_idx]
    else:
        patient_dirs = patient_dirs[val_split_idx:]

    data = [{
        "image": str(image_path),
        "label": labels_df.loc[image_path.parent.name, "abnormal"],
    } for image_path in image_paths]

    return data


class SinoCTDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        transform: Callable = None,
        subset: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),  # train, val, test
        seed: int = 0,
    ):
        data = _initialize_dataset(data_dir, subset, split_ratio, seed)
        super().__init__(data=data, transform=transform)


class SinoCTPersistentDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str, 
        cache_dir: str,
        transform: Callable = None,
        subset: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),  # train, val, test
        seed: int = 0,
    ):
        data = _initialize_dataset(data_dir, subset, split_ratio, seed)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class SinoCTGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str, 
        cache_dir: str,
        device: int,
        transform: Callable = None,
        subset: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),  # train, val, test
        seed: int = 0,
    ):
        data = _initialize_dataset(data_dir, subset, split_ratio, seed)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
