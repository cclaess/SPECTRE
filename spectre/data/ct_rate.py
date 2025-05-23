import os
from pathlib import Path
from typing import Callable

from monai.data import Dataset

from spectre.data.cache_dataset import CacheDataset
from spectre.data.gds_dataset import GDSDataset


class CTRateDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train"
    ):
        image_paths = Path(data_dir).glob(os.path.join('dataset', subset, "*", "*", "*.nii.gz"))
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

        super().__init__(data=data, transform=transform)


class CTRateCacheDataset(CacheDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train"
    ):
        image_paths = Path(data_dir).glob(os.path.join('dataset', subset, "*", "*", "*.nii.gz"))
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
    ):
        image_paths = Path(data_dir).glob(os.path.join('dataset', subset, "*", "*", "*.nii.gz"))
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

        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
