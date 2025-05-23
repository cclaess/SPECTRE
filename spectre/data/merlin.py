import os
from pathlib import Path
from typing import Callable

import pandas as pd
from monai.data import Dataset

from spectre.data.cache_dataset import CacheDataset
from spectre.data.gds_dataset import GDSDataset


def parse_name(image_path):
    return image_path.name.replace(".nii.gz", "")


class MerlinDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train"
    ):
        image_paths = Path(data_dir).glob(os.path.join("merlinabdominalctdataset", "merlin_data", "*.nii.gz"))
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

        super().__init__(data=data, transform=transform)


class MerlinCacheDataset(CacheDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train"
    ):
        image_paths = Path(data_dir).glob(os.path.join("merlinabdominalctdataset", "merlin_data", "*.nii.gz"))
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
    ):
        image_paths = Path(data_dir).glob(os.path.join("merlinabdominalctdataset", "merlin_data", "*.nii.gz"))
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

        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
