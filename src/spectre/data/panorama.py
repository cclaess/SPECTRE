from pathlib import Path
from typing import Callable, List, Dict

from monai.data import Dataset

from spectre.data._base_datasets import PersistentDataset, GDSDataset


def _initialize_dataset(
    data_dir: str,
    fraction: float = 1.0,
) -> List[Dict[str, str]]:
    image_paths = sorted(Path(data_dir).glob("*.nii.gz"))

    if 0. < fraction < 1.0:
        n_keep = int(len(image_paths) * fraction)
        image_paths = image_paths[:n_keep]

    data = [{"image": str(image_path)} for image_path in image_paths]
    return data


class PanoramaDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        transform: Callable = None,
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, fraction=fraction)
        super().__init__(data=data, transform=transform)


class PanoramaPersistentDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        transform: Callable = None,
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, fraction=fraction)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class PanoramaGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        transform: Callable = None,
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, fraction=fraction)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
