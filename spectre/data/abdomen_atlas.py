import os
from pathlib import Path
from typing import Callable, Dict, List

from monai.data import Dataset

from spectre.data.cache_dataset import CacheDataset
from spectre.data.gds_dataset import GDSDataset


def _initialize_dataset(
    data_dir: str, 
    include_labels: bool = False
) -> List[Dict[str, str]]:
    
    image_paths = Path(data_dir).glob(os.path.join("*", "ct.nii.gz"))

    if include_labels:
        label_paths = Path(data_dir).glob(os.path.join("*", "combined_labels.nii.gz"))

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
        transform: Callable = None
    ):
        data = _initialize_dataset(data_dir, include_labels)
        super().__init__(data=data, transform=transform)


class AbdomenAtlasCacheDataset(CacheDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_labels: bool = False, 
        transform: Callable = None
    ):
        data = _initialize_dataset(data_dir, include_labels)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class AbdomenAtlasGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        include_labels: bool = False, 
        transform: Callable = None,
    ):
        data = _initialize_dataset(data_dir, include_labels)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
