import os
from pathlib import Path
from typing import Callable

from monai.data import Dataset, PersistentDataset


class CTRateDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        include_reports: bool = False, 
        transform: Callable = None
    ):
        image_paths = Path(data_dir).glob(os.path.join("dataset", "train", "*", "*", "*.nii.gz"))

        if include_reports:
            import pandas as pd
            reports = pd.read_csv(os.path.join(
                data_dir, "dataset", "radiology_text_reports", "train_reports.csv"
            ))

            data = [{
                "image": str(image_path),
                "report": reports[reports["VolumeName"] == image_path.name]["Findings_EN"].values[0]
            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)


class CTRateCacheDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_reports: bool = False, 
        transform: Callable = None
    ):
        image_paths = Path(data_dir).glob(os.path.join("dataset", "train", "*", "*", "*.nii.gz"))

        if include_reports:
            import pandas as pd
            reports = pd.read_csv(os.path.join(
                data_dir, "dataset", "radiology_text_reports", "train_reports.csv"
            ))

            data = [{
                "image": str(image_path),
                "report": reports[reports["VolumeName"] == image_path.name]["Findings_EN"].values[0]
            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
