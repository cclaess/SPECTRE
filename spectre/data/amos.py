import os
from pathlib import Path
from typing import Callable

from monai.data import Dataset

from spectre.data.cache_dataset import CacheDataset
from spectre.data.gds_dataset import GDSDataset


class AmosDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        transform: Callable = None
    ):
        image_paths = Path(data_dir).glob(os.path.join("*", "*", "amos_*.nii.gz"))
        data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)


class AmosCacheDataset(CacheDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        transform: Callable = None
    ):
        image_paths = Path(data_dir).glob(os.path.join("*", "*", "amos_*.nii.gz"))
        data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
        

class AmosGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        transform: Callable = None,
    ):
        image_paths = Path(data_dir).glob(os.path.join("*", "*", "amos_*.nii.gz"))
        data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
