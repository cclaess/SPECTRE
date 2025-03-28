import os
import random
from typing import Union, Callable, Optional, List

import torch
from torch.utils.data import ConcatDataset
from monai.data import DataLoader
from monai.data.utils import list_data_collate


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
        # elif dataset == "merlin":
        #     kwargs = {
        #         "data_dir": os.path.join(data_dir, "MERLIN"),
        #         "include_reports": include_reports,
        #         "transform": transform,
        #     }
        #     if cache_dataset:
        #         from spectre.data import MerlinCacheDataset
        #         datasets_list.append(MerlinCacheDataset(**kwargs, cache_dir=cache_dir))
        #     else:
        #         from spectre.data import MerlinDataset
        #         datasets_list.append(MerlinDataset(**kwargs))

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


def extended_collate(samples_list, mask_ratio=None, mask_probability=None, n_tokens=None, mask_generator=None):
    """
    Applies MONAI's list_data_collate first and then extends it with DINOv2 masking logic.

    Args:
        samples_list: List of samples containing 'global_crops' and 'local_crops'.
        mask_ratio: Tuple defining the range of masking ratios.
        mask_probability: Probability of applying masking.
        dtype: Data type to cast the collated tensors.
        n_tokens: Number of tokens for masking.
        mask_generator: Function to generate masks.

    Returns:
        A dictionary with collated global/local crops and corresponding masks.
    """
    # Apply MONAI's list_data_collate
    collated_data = list_data_collate(samples_list)

    # Extract crops
    global_crops = torch.cat(collated_data["global_crops"], dim=0)
    local_crops = torch.cat(collated_data["local_crops"], dim=0)

    if (
        mask_ratio is None
        or mask_probability is None 
        or n_tokens is None 
        or mask_generator is None
    ):
        return {
            "global_crops": global_crops,
            "local_crops": local_crops,
        }
    
    else:
        # Masking logic (DINOv2 style)
        B = len(global_crops)
        N = n_tokens
        n_samples_masked = int(B * mask_probability)

        probs = torch.linspace(*mask_ratio, n_samples_masked + 1)
        upperbound: int = 0
        masks_list = []

        for i in range(n_samples_masked):
            prob_min, prob_max = probs[i], probs[i + 1]
            masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)

        for _ in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(mask_generator(0)))

        random.shuffle(masks_list)
        collated_masks = torch.stack(masks_list).flatten(1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

        masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

        return {
            "global_crops": global_crops,
            "local_crops": local_crops,
            "masks": collated_masks,
            "mask_indices": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
        }
