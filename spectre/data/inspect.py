import os
from pathlib import Path
from typing import Callable

from monai.data import Dataset

from spectre.data.cache_dataset import CacheDataset
from spectre.data.gds_dataset import GDSDataset


def parse_name(image_path):
    return image_path.name.replace(".nii.gz", "")


class InspectDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
    ):
        image_paths = Path(data_dir).glob(os.path.join('inspect2', "CTPA", "*.nii.gz"))
        if include_reports:
            import pandas as pd
            text_path = os.path.join(Path(data_dir), "inspect2", "Final_Impressions.xlsx")
            reports = pd.read_excel(text_path)

            data = [{
                "image": str(image_path),
                "impressions": [
                    val for val in [
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_EN"].values[0],
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_1"].values[0],
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_2"].values[0]
                    ] if isinstance(val, str)
                ]
            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)


class InspectCacheDataset(CacheDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
    ):
        image_paths = Path(data_dir).glob(os.path.join('inspect2', "CTPA", "*.nii.gz"))
        if include_reports:
            import pandas as pd
            text_path = os.path.join(Path(data_dir), "inspect2", "Final_Impressions.xlsx")
            reports = pd.read_excel(text_path)

            data = [{
                "image": str(image_path),
                "impressions": [
                    val for val in [
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_EN"].values[0],
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_1"].values[0],
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_2"].values[0]
                    ] if isinstance(val, str)
                ]
            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class InspectGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        include_reports: bool = False, 
        transform: Callable = None,
    ):
        image_paths = Path(data_dir).glob(os.path.join('inspect2', "CTPA", "*.nii.gz"))
        if include_reports:
            import pandas as pd
            text_path = os.path.join(Path(data_dir), "inspect2", "Final_Impressions.xlsx")
            reports = pd.read_excel(text_path)

            data = [{
                "impressions": [
                    val for val in [
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_EN"].values[0],
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_1"].values[0],
                        reports[reports["impression_id"] == parse_name(image_path)]["Impressions_2"].values[0]
                    ] if isinstance(val, str)
                ]
            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
