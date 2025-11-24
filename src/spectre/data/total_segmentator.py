import os
from pathlib import Path
from typing import Callable, List, Union, Dict

import pandas as pd
from monai.data import Dataset

from spectre.data._base_datasets import PersistentDataset, GDSDataset


LABEL_GROUPS = {
    "cardiac": [
        "heart", "aorta", "pulmonary_vein", "brachiocephalic_trunk", "subclavian_artery_left",
        "subclavian_artery_right", "common_carotid_artery_left", "common_carotid_artery_right",
        "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left",
        "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein",
        "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right",
    ],
    "muscles": [
        "humerus_left", "humerus_right", "fibula",  # not found in the dataset
        "tibia",  # not found in the dataset
        "femur_left", "femur_right", "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left",
        "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left",
        "iliopsoas_right", "quadriceps_femoris_left",  # not found in the dataset
        "quadriceps_femoris_right",  # not found in the dataset
        "thigh_medial_compartment_left",  # not found in the dataset
        "thigh_medial_compartment_right",  # not found in the dataset
        "thigh_posterior_compartment_left",  # not found in the dataset
        "thigh_posterior_compartment_right",  # not found in the dataset
        "sartorius_left",  # not found in the dataset
        "sartorius_right",  # not found in the dataset
        "brain",
    ],
    "organs": [
        "spleen", "kidney_left", "kidney_right", "gallbladder", "liver", "stomach",
        "pancreas", "adrenal_gland_left", "adrenal_gland_right", "lung_lower_lobe_left",
        "lung_lower_lobe_right", "lung_middle_lobe_right", "lung_upper_lobe_left",
        "lung_upper_lobe_right", "esophagus", "trachea", "thyroid_gland", "small_bowel",
        "duodenum", "colon", "urinary_bladder", "prostate", "kidney_cyst_left",
        "kidney_cyst_right",
    ],
    "ribs": [
        "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
        "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
        "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
        "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", 
        "rib_right_12", "sternum", "costal_cartilages",
    ],
    "vertebrae": [
        "sacrum", "vertebrae_S1", "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4",
        "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9",
        "vertebrae_T10", "vertebrae_T11", "vertebrae_T12", "vertebrae_L1", "vertebrae_L2",
        "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_C1", "vertebrae_C2",
        "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    ],
    "lungs": [
        "lung_lower_lobe_left", "lung_lower_lobe_right", "lung_middle_lobe_right", 
        "lung_upper_lobe_left", "lung_upper_lobe_right"
    ]
}

def _initialize_dataset(
    data_dir: str,
    include_labels: bool = False,
    label_groups: Union[str, List[str]] = [
        "Cardiac", "Muscles", "Organs", "Ribs", "Vertebrae"
    ],
    subset: str = "train",
) -> List[Dict[str, str]]:
    
    image_paths = Path(data_dir).glob(os.path.join("*", "ct.nii.gz"))

    # Filter metadata for the specified subset
    meta = pd.read_csv(os.path.join(data_dir, "meta.csv"), sep=";")
    if subset not in ["train", "val", "test"]:
        raise ValueError(f"Invalid subset: {subset}. Choose from 'train', 'val', or 'test'.")
    meta = meta[meta["split"] == subset]
    image_ids = meta["image_id"].tolist()
    image_paths = [path for path in image_paths if path.parent.name in image_ids]

    if include_labels:
        label_groups = list(label_groups)
        assert all(
            group.lower() in LABEL_GROUPS for group in label_groups
        ), f"Invalid label group(s): {label_groups}"

        data = []
        for image_path in image_paths:
            seg_dir = image_path.parent / "segmentations"

            if not seg_dir.exists():
                raise FileNotFoundError(
                    f"Segmentations directory not found for {image_path}: {seg_dir}"
                )

            data_sample = {"image": str(image_path)}
            for group in label_groups:
                group_labels = LABEL_GROUPS[group.lower()]
                for label in group_labels:
                    label_path = seg_dir / f"{label}.nii.gz"
                    if label_path.exists():
                        data_sample[label] = str(label_path)
                    else:
                        print(f"Warning: Label {label} not found for {image_path}")
            data.append(data_sample)
    else:
        data = [{"image": str(image_path)} for image_path in image_paths]

    return data


class TotalSegmentatorDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        include_labels: bool = False,
        label_groups: Union[str, List[str]] = [
            "Cardiac", "Muscles", "Organs", "Ribs", "Vertebrae"
        ],
        transform: Callable = None,
        subset: str = "train"
    ):
        data = _initialize_dataset(data_dir, include_labels, label_groups, subset)
        super().__init__(data=data, transform=transform)


class TotalSegmentatorPersistentDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        include_labels: bool = False, 
        label_groups: Union[str, List[str]] = [
            "Cardiac", "Muscles", "Organs", "Ribs", "Vertebrae"
        ],
        subset: str = "train",
        transform: Callable = None
    ):
        data = _initialize_dataset(data_dir, include_labels, label_groups, subset)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)


class TotalSegmentatorGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        include_labels: bool = False, 
        label_groups: Union[str, List[str]] = [
            "Cardiac", "Muscles", "Organs", "Ribs", "Vertebrae"
        ],
        subset: str = "train",
        transform: Callable = None,
    ):
        data = _initialize_dataset(data_dir, include_labels, label_groups, subset)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
