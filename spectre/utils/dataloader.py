from typing import Callable, Optional

from monai.data import DataLoader


def get_dataloader(
    dataset: str,
    dataset_path: str,
    include_reports: bool = False,
    cache_dataset: bool = False,
    transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Get dataloader for training.
    """
    if dataset == "ct_rate":
        if not cache_dataset:
            from spectre.data import CTRateDataset
            dataset = CTRateDataset(
                dataset_path, 
                include_reports=include_reports,
                transform=transform,
            )
        else:
            from spectre.data import CTRateCacheDataset
            dataset = CTRateCacheDataset(
                dataset_path, 
                include_reports=include_reports,
                transform=transform,
            )

    elif dataset == "nlst":
        if not cache_dataset:
            from spectre.data import NLSTDataset
            dataset = NLSTDataset(
                dataset_path,
                transform=transform,
            )
        else:
            from spectre.data import NLSTCacheDataset
            dataset = NLSTCacheDataset(
                dataset_path,
                transform=transform,
            )

    elif dataset == "nlst+ct_rate":
        if not cache_dataset:
            from spectre.data import NLSTCTRateDataset
            dataset = NLSTCTRateDataset(
                dataset_path,
                transform=transform,
            )
        else:
            from spectre.data import NLSTCTRateCacheDataset
            dataset = NLSTCTRateCacheDataset(
                dataset_path,
                transform=transform,
            )

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return dataloader