import os
from pathlib import Path
from typing import Callable

from monai.data import Dataset, CacheDataset


class CTRateDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        include_reports: bool = False, 
        transform: Callable = None
    ):
        series_folders = Path(dataset_path).glob(os.path.join("dataset", "train", "*", "*"))
        reconstruction_paths_per_series = [image.glob("*.nii.gz") for image in series_folders]

        if include_reports:
            import pandas as pd
            reports = pd.read_csv(os.path.join(
                dataset_path, "dataset", "radiology_text_reports", "train_reports.csv"
            ))

            data = [{
                "image": [str(image) for image in series],
                "report": reports[reports["VolumeName"] == series[0].name]["Findings_EN"].values[0]
            } for series in reconstruction_paths_per_series]
        else:
            data = [{"image": [str(image) for image in series]} for series in reconstruction_paths_per_series]

        super().__init__(data=data, transform=transform)


class CTRateCacheDataset(CacheDataset):
    def __init__(
        self, 
        dataset_path: str, 
        include_reports: bool = False, 
        transform: Callable = None
    ):
        series_folders = Path(dataset_path).glob(os.path.join("dataset", "train", "*", "*"))
        reconstruction_paths_per_series = [image.glob("*.nii.gz") for image in series_folders]

        if include_reports:
            import pandas as pd
            reports = pd.read_csv(os.path.join(
                dataset_path, "dataset", "radiology_text_reports", "train_reports.csv"
            ))

            data = [{
                "image": [str(image) for image in series],
                "report": reports[reports["VolumeName"] == series[0].name]["Findings_EN"].values[0]
            } for series in reconstruction_paths_per_series]
        else:
            data = [{"image": [str(image) for image in series]} for series in reconstruction_paths_per_series]

        super().__init__(data=data, transform=transform)
