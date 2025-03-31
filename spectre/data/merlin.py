import os
import pandas as pd
from pathlib import Path
from typing import Callable
from monai.data import Dataset, PersistentDataset
import torch

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
        image_paths = Path(data_dir).glob(os.path.join( "dataset", subset, "*", "*.nii.gz"))
        if include_reports:
            import pandas as pd
            text_path = os.path.join(Path(data_dir),  "dataset", "reports.xlsx" )
            reports = pd.read_excel(text_path)

            data = [{
                "image": str(image_path),
                "findings": [reports[reports["study id"] == parse_name(image_path)]["Findings_0"].values[0],
                reports[reports["study id"] == parse_name(image_path)]["Findings_1"].values[0],
                reports[reports["study id"] == parse_name(image_path)]["Findings_2"].values[0]],

                "impressions": [reports[reports["study id"] == parse_name(image_path)]["Impressions_0"].values[0],
                reports[reports["study id"] == parse_name(image_path)]["Impressions_1"].values[0],
                reports[reports["study id"] == parse_name(image_path)]["Impressions_2"].values[0]],

                "icd10": reports[reports["study id"] == parse_name(image_path)]["FULL_ICD10 Description"].values[0]

            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)

class MerlinCacheDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train"
    ):
        image_paths = Path(data_dir).glob(os.path.join( "dataset", subset, "*", "*.nii.gz"))
        if include_reports:
            import pandas as pd
            text_path = os.path.join(Path(data_dir),  "dataset", "reports.xlsx" )
            reports = pd.read_excel(text_path)

            data = [{
                "image": str(image_path),
                "findings": [reports[reports["study id"] == parse_name(image_path)]["Findings_0"].values[0],
                reports[reports["study id"] == parse_name(image_path)]["Findings_1"].values[0],
                reports[reports["study id"] == parse_name(image_path)]["Findings_2"].values[0]],

                "impressions": [reports[reports["study id"] == parse_name(image_path)]["Impressions_0"].values[0],
                reports[reports["study id"] == parse_name(image_path)]["Impressions_1"].values[0],
                reports[reports["study id"] == parse_name(image_path)]["Impressions_2"].values[0]],

                "icd10": reports[reports["study id"] == parse_name(image_path)]["FULL_ICD10 Description"].values[0]

            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir)