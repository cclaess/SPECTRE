import os
import pandas as pd
from pathlib import Path
from typing import Callable
from monai.data import Dataset, PersistentDataset
import torch

class CTRatePeristentDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train"
    ):
        image_paths = Path(data_dir).glob(os.path.join("dataset", subset, "*", "*", "*.nii.gz"))

        if include_reports:
            import pandas as pd
            reports = pd.read_csv(os.path.join(
                data_dir, "dataset", "radiology_text_reports", "train_reports.csv"
            ))

            data = [{
                "image": str(image_path),
                "findings_0": reports[reports["VolumeName"] == image_path.name]["findings_0"].values[0],
                "findings_1": reports[reports["VolumeName"] == image_path.name]["findings_1"].values[0],
                "findings_2": reports[reports["VolumeName"] == image_path.name]["findings_2"].values[0],

                "impressions_0": reports[reports["VolumeName"] == image_path.name]["impressions_0"].values[0],
                "impressions_1": reports[reports["VolumeName"] == image_path.name]["impressions_1"].values[0],
                "impressions_2": reports[reports["VolumeName"] == image_path.name]["impressions_2"].values[0],

            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

def clean_ipath_to_name(image_path):
    return image_path.name.replace(".nii.gz", "")

class CTRateDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train"
    ):
        image_paths = Path(data_dir).glob(os.path.join("CT-RATE", subset, "*", "*", "*.nii.gz"))
        if include_reports:
            import pandas as pd
            text_path = os.path.join(Path(data_dir),"CT-RATE","radiology_text_reports", f"{subset}_reports.xlsx" )
            reports = pd.read_excel(text_path)

            data = [{
                "image": str(image_path),
                "findings": [reports[reports["VolumeName"] == image_path.name]["Findings_EN"].values[0],
                reports[reports["VolumeName"] == image_path.name]["Findings_1"].values[0],
                reports[reports["VolumeName"] == image_path.name]["Findings_2"].values[0]],

                "impressions": [reports[reports["VolumeName"] == image_path.name]["Impressions_EN"].values[0],
                reports[reports["VolumeName"] == image_path.name]["Impressions_1"].values[0],
                reports[reports["VolumeName"] == image_path.name]["Impressions_2"].values[0]],

            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)

class MerlinDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        include_reports: bool = False, 
        transform: Callable = None,
        subset: str = "train"
    ):
        image_paths = Path(data_dir).glob(os.path.join("MERLIN", "dataset", subset, "*", "*.nii.gz"))
        if include_reports:
            import pandas as pd
            text_path = os.path.join(Path(data_dir), "MERLIN", "dataset", "reports.xlsx" )
            reports = pd.read_excel(text_path)

            data = [{
                "image": str(image_path),
                "findings": [reports[reports["study id"] == clean_ipath_to_name(image_path)]["Findings_0"].values[0],
                reports[reports["study id"] == clean_ipath_to_name(image_path)]["Findings_1"].values[0],
                reports[reports["study id"] == clean_ipath_to_name(image_path)]["Findings_2"].values[0]],

                "impressions": [reports[reports["study id"] == clean_ipath_to_name(image_path)]["Impressions_0"].values[0],
                reports[reports["study id"] == clean_ipath_to_name(image_path)]["Impressions_1"].values[0],
                reports[reports["study id"] == clean_ipath_to_name(image_path)]["Impressions_2"].values[0]],

                "icd10": reports[reports["study id"] == clean_ipath_to_name(image_path)]["FULL_ICD10 Description"].values[0]

            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)


class VisionLanguageDataset(Dataset):
    # Combines the Merlin and CTRate datasets
    def __init__(self, merlin_data=None, ctrate_data=None,
                    transform=None):    
        self.merlin_data = merlin_data
        self.ctrate_data = ctrate_data

        self.data = torch.utils.data.ConcatDataset([self.merlin_data, self.ctrate_data])

        super().__init__(data=self.data, transform=transform)
    




    
        

