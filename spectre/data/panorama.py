from pathlib import Path
from typing import Callable, List, Dict

from monai.data import Dataset

from spectre.data.cache_dataset import CacheDataset
from spectre.data.gds_dataset import GDSDataset


def _initialize_dataset(data_dir: str) -> List[Dict[str, str]]:
    image_paths = Path(data_dir).glob("*.nii.gz")
    data = [{"image": str(image_path)} for image_path in image_paths]
    return data


class PanoramaDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        transform: Callable = None
    ):
        data = _initialize_dataset(data_dir)
        super().__init__(data=data, transform=transform)


class PanoramaCacheDataset(CacheDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        transform: Callable = None
    ):
        data = _initialize_dataset(data_dir)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class PanoramaGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        transform: Callable = None,
    ):
        data = _initialize_dataset(data_dir)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
