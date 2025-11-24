import os
from pathlib import Path
from typing import Callable, List, Dict

import pandas as pd
from monai.data import Dataset

from spectre.data._base_datasets import PersistentDataset, GDSDataset

def parse_name(image_path):
    return image_path.name.replace(".nii.gz", "")


def _initialize_dataset(
    data_dir: str,
    include_reports: bool = False,
    subset: str = "train",
    fraction: float = 1.0,
) -> List[Dict[str, str]]:

    image_paths = sorted(Path(data_dir).glob(os.path.join(
        "merlinabdominalctdataset", "merlin_data", "*.nii.gz")))

    if 0. < fraction < 1.0:
        n_keep = int(len(list(image_paths)) * fraction)
        image_paths = image_paths[:n_keep]

    text_path = Path(data_dir) / "merlinabdominalctdataset" / "reports_final_updated.xlsx"
    reports = pd.read_excel(text_path)
    image_paths = [p for p in image_paths if \
        reports[reports["study id"] == parse_name(p)]["Split"].values[0] == subset]
    
    if include_reports:
        if subset == "train":
            data = [{
                "image": str(image_path),
                "findings": [
                    val for val in [
                        reports[reports["study id"] == parse_name(image_path)]["Findings_EN"].values[0],
                        reports[reports["study id"] == parse_name(image_path)]["Findings_1"].values[0],
                        reports[reports["study id"] == parse_name(image_path)]["Findings_2"].values[0]
                    ] if isinstance(val, str)
                ],
                "impressions": [
                    val for val in [
                        reports[reports["study id"] == parse_name(image_path)]["Impressions_EN"].values[0],
                        reports[reports["study id"] == parse_name(image_path)]["Impressions_1"].values[0],
                        reports[reports["study id"] == parse_name(image_path)]["Impressions_2"].values[0]
                    ] if isinstance(val, str)
                ],
                "icd10": reports[reports["study id"] == parse_name(image_path)]["FULL ICD10 Cleaned"].values[0]
            } for image_path in image_paths]

        else:
            data = [{
                "image": str(image_path),
                "findings": [reports[reports["study id"] == parse_name(image_path)]["Findings_EN"].values[0]],
                "impressions": [reports[reports["study id"] == parse_name(image_path)]["Impressions_EN"].values[0]],

                "icd10": reports[reports["study id"] == parse_name(image_path)]["FULL ICD10 Cleaned"].values[0]

            } for image_path in image_paths]
    else:
        data = [{"image": str(image_path)} for image_path in image_paths]
    return data


class MerlinDataset(Dataset):
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


class MerlinPersistentDataset(PersistentDataset):
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


class MerlinGDSDataset(GDSDataset):
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
