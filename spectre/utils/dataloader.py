import os
from typing import Union, Callable, Optional, List

from torch.utils.data import ConcatDataset
from monai.data import DataLoader


def get_dataloader(
    datasets: Union[str, List[str]],
    data_dir: str,
    include_reports: bool = False,
    include_labels: bool = False,
    cache_dataset: bool = False,
    cache_dir: Optional[str] = None,
    transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Get dataloader for training.
    """

    if isinstance(datasets, str):
        datasets = [datasets]

    if include_reports:
        assert set(datasets).issubset({"ct_rate", "merlin"}), (
            "When include_reports is True, only 'ct_rate' and 'merlin' datasets are allowed.")
    if include_labels:
        assert set(datasets).issubset({"abdomen_atlas", "abdomenct_1k"}), (
            "When include_labels is True, only 'abdomen_atlas' and 'abdomenct_1k' datasets are allowed.")
    
    datasets_list = []
    for dataset in datasets:
        # CT-RATE dataset with paired chest CT and radiology reports
        if dataset == "ct_rate":
            kwargs = {
                "data_dir": os.path.join(data_dir, "CT-RATE"),
                "include_reports": include_reports,
                "transform": transform,
            }
            if cache_dataset:
                from spectre.data import CTRateCacheDataset
                datasets_list.append(CTRateCacheDataset(**kwargs, cache_dir=cache_dir))
            else:
                from spectre.data import CTRateDataset
                datasets_list.append(CTRateDataset(**kwargs))

        # INSPECT dataset with paired chest CTPA and radiology reports
        # elif dataset == "inspect":
        #     kwargs = {
        #         "data_dir": os.path.join(data_dir, "INSPECT"),
        #         "include_reports": include_reports,
        #         "transform": transform,
        #     }
        #     if cache_dataset:
        #         from spectre.data import InspectCacheDataset
        #         datasets_list.append(InspectCacheDataset(**kwargs, cache_dir=cache_dir))
        #     else:
        #         from spectre.data import InspectDataset
        #         datasets_list.append(InspectDataset(**kwargs))

        # MERLIN dataset with paired abdominal CT and radiology reports
        elif dataset == "merlin":
            kwargs = {
                "data_dir": os.path.join(data_dir, "MERLIN"),
                "include_reports": include_reports,
                "transform": transform,
            }
            if cache_dataset:
                from spectre.data import MerlinCacheDataset
                datasets_list.append(MerlinCacheDataset(**kwargs, cache_dir=cache_dir))
            else:
                from spectre.data import MerlinDataset
                datasets_list.append(MerlinDataset(**kwargs))

        # NLST dataset with low-dose chest CT
        elif dataset == "nlst":
            kwargs = {
                "data_dir": os.path.join(data_dir, "NLST"),
                "transform": transform,
            }
            if cache_dataset:
                from spectre.data import NlstCacheDataset
                datasets_list.append(NlstCacheDataset(**kwargs, cache_dir=cache_dir))
            else:
                from spectre.data import NlstDataset
                datasets_list.append(NlstDataset(**kwargs))
        
        # Amos dataset with abdominal CT
        elif dataset == "amos":
            kwargs = {
                "data_dir": os.path.join(data_dir, "Amos"),
                "transform": transform,
            }
            if cache_dataset:
                from spectre.data import AmosCacheDataset
                datasets_list.append(AmosCacheDataset(**kwargs, cache_dir=cache_dir))
            else:
                from spectre.data import AmosDataset
                datasets_list.append(AmosDataset(**kwargs))
        
        # AbdomenAtlas 1.0 Mini dataset with abdominal CT and organ segmentations
        elif dataset == "abdomen_atlas":
            kwargs = {
                "data_dir": os.path.join(data_dir, "AbdomenAtlas1.0Mini"),
                "include_labels": include_labels,
                "transform": transform,
            }
            if cache_dataset:
                from spectre.data import AbdomenAtlasCacheDataset
                datasets_list.append(AbdomenAtlasCacheDataset(**kwargs, cache_dir=cache_dir))
            else:
                from spectre.data import AbdomenAtlasDataset
                datasets_list.append(AbdomenAtlasDataset(**kwargs))
        
        # Panorama dataset with abdominal contrast-enhanced CT
        elif dataset == "panorama":
            kwargs = {
                "data_dir": os.path.join(data_dir, "PANORAMA"),
                "transform": transform,
            }
            if cache_dataset:
                from spectre.data import PanoramaCacheDataset
                datasets_list.append(PanoramaCacheDataset(**kwargs, cache_dir=cache_dir))
            else:
                from spectre.data import PanoramaDataset
                datasets_list.append(PanoramaDataset(**kwargs))
        
        # AbdomenCT-1k dataset with abdominal CT and organ segmentations
        elif dataset == "abdomenct_1k":
            kwargs = {
                "data_dir": os.path.join(data_dir, "AbdomenCT-1K"),
                "include_labels": include_labels,
                "transform": transform,
            }
            if cache_dataset:
                from spectre.data import AbdomenCT1KCacheDataset
                datasets_list.append(AbdomenCT1KCacheDataset(**kwargs, cache_dir=cache_dir))
            else:
                from spectre.data import AbdomenCT1kDataset
                datasets_list.append(AbdomenCT1kDataset(**kwargs))

        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented.")
    
    if len(datasets_list) == 0:
        raise ValueError("No datasets found. Please check the dataset names.")
    elif len(datasets_list) == 1:
        dataset = datasets_list[0]
    else:
        dataset = ConcatDataset(datasets_list)
    
    if collate_fn is not None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )  # Cannot pass None to use MONAI collate_fn
    return dataloader
