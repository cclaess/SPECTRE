import random
from typing import Callable, Optional

import torch
from monai.data import DataLoader
from monai.data.utils import list_data_collate


def get_dataloader(
    dataset: str,
    data_dir: str,
    include_reports: bool = False,
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
    if dataset == "ct_rate":
        if not cache_dataset:
            from spectre.data import CTRateDataset
            dataset = CTRateDataset(
                data_dir, 
                include_reports=include_reports,
                transform=transform,
            )
        else:
            from spectre.data import CTRateCacheDataset
            dataset = CTRateCacheDataset(
                data_dir,
                cache_dir,
                include_reports=include_reports,
                transform=transform,
            )

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    
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


def extended_collate(samples_list, mask_ratio, mask_probability=None, n_tokens=None, mask_generator=None):
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

    if (mask_probability is None or n_tokens is None or mask_generator is None):
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
