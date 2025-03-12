from typing import Callable, Optional

from monai.data import DataLoader


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
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return dataloader
