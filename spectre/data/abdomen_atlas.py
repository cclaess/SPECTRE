import os
from pathlib import Path
from typing import Callable, Dict, List

from monai.data import Dataset

from spectre.data._base_datasets import PersistentDataset, GDSDataset


def _initialize_dataset(
    data_dir: str, 
    include_labels: bool = False,
    fraction: float = 1.0,
) -> List[Dict[str, str]]:

    image_paths = sorted(Path(data_dir).glob(os.path.join("*", "ct.nii.gz")))
    
    if 0. < fraction < 1.0:
        n_keep = int(len(image_paths) * fraction)
        image_paths = image_paths[:n_keep]

    if include_labels:
        label_paths = sorted(Path(data_dir).glob(os.path.join("*", "combined_labels.nii.gz")))
        if 0. < fraction < 1.0:
            label_paths = label_paths[:n_keep]

        data = [{
            "image": str(image_path),
            "label": str(label_path)
        } for image_path, label_path in zip(image_paths, label_paths)]
    else:
        data = [{"image": str(image_path)} for image_path in image_paths]

    return data


class AbdomenAtlasDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        include_labels: bool = False, 
        transform: Callable = None,
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, include_labels, fraction)
        super().__init__(data=data, transform=transform)


class AbdomenAtlasPersistentDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_labels: bool = False, 
        transform: Callable = None,
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, include_labels, fraction)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class AbdomenAtlasGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        include_labels: bool = False, 
        transform: Callable = None,
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, include_labels, fraction)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
