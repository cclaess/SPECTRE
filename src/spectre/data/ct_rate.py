import os
from pathlib import Path
from typing import Callable, Dict, List

from monai.data import Dataset

from spectre.data._base_datasets import PersistentDataset, GDSDataset

def _initialize_dataset(
    data_dir: str,
    include_reports: bool = False,
    subset: str = "train",
    fraction: float = 1.0,
) -> List[Dict[str, str]]:
    
    image_paths = sorted(Path(data_dir).glob(os.path.join('dataset', subset, "*", "*", "*.nii.gz")))

    if 0. < fraction < 1.0:
        n_keep = int(len(image_paths) * fraction)
        image_paths = image_paths[:n_keep]

    if include_reports:
        import pandas as pd
        text_path = os.path.join(Path(data_dir), 'dataset', "radiology_text_reports", f"{subset}_reports.xlsx" )
        reports = pd.read_excel(text_path)
        if subset == "train":
            data = [{
                "image": str(image_path),
                "findings": [val for val in [
                    reports[reports["VolumeName"] == image_path.name]["Findings_EN"].values[0],
                    reports[reports["VolumeName"] == image_path.name]["Findings_1"].values[0],
                    reports[reports["VolumeName"] == image_path.name]["Findings_2"].values[0]
                ] if isinstance(val, str)],
                "impressions": [val for val in [
                    reports[reports["VolumeName"] == image_path.name]["Impressions_EN"].values[0],
                    reports[reports["VolumeName"] == image_path.name]["Impressions_1"].values[0],
                    reports[reports["VolumeName"] == image_path.name]["Impressions_2"].values[0]
                ] if isinstance(val, str)],
            } for image_path in image_paths]
        else:
            data = [{
                "image": str(image_path),
                "findings": [reports[reports["VolumeName"] == image_path.name]["Findings_EN"].values[0]],

                "impressions": [reports[reports["VolumeName"] == image_path.name]["Impressions_EN"].values[0]],

            } for image_path in image_paths]
    else:
        data = [{"image": str(image_path)} for image_path in image_paths]
    return data


class CTRateDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train",
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, include_reports, subset, fraction)
        super().__init__(data=data, transform=transform)


class CTRatePersistentDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train",
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, include_reports, subset, fraction)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class CTRateGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train",
        fraction: float = 1.0,
    ):
        data = _initialize_dataset(data_dir, include_reports, subset, fraction)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
