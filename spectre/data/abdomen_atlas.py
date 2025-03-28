import os
from pathlib import Path
from typing import Callable

from monai.data import Dataset, PersistentDataset


class AbdomenAtlasDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        include_labels: bool = False, 
        transform: Callable = None
    ):
        image_paths = Path(data_dir).glob(os.path.join("*", "ct.nii.gz"))

        if include_labels:
            label_paths = Path(data_dir).glob(os.path.join("*", "combined_labels.nii.gz"))

            data = [{
                "image": str(image_path),
                "label": str(label_path)
            } for image_path, label_path in zip(image_paths, label_paths)]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)


class AbdomenAtlasCacheDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_labels: bool = False, 
        transform: Callable = None
    ):
        image_paths = Path(data_dir).glob(os.path.join("*", "ct.nii.gz"))

        if include_labels:
            label_paths = Path(data_dir).glob(os.path.join("*", "combined_labels.nii.gz"))

            data = [{
                "image": str(image_path),
                "label": str(label_path)
            } for image_path, label_path in zip(image_paths, label_paths)]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
