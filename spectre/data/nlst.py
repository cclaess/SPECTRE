import os
from pathlib import Path
from typing import Callable

from monai.data import Dataset

from spectre.data.cache_dataset import CacheDataset


class NlstDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        transform: Callable = None
    ):
        image_paths = Path(data_dir).glob(os.path.join("*", "*", "*.nii.gz"))
        data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)


class NlstCacheDataset(CacheDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        transform: Callable = None
    ):
        image_paths = Path(data_dir).glob(os.path.join("*", "*", "*.nii.gz"))
        data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
