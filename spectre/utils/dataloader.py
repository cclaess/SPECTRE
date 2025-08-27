import os
from typing import Union, Callable, Optional, List

import torch
import monai.data as data
from torch.utils.data import ConcatDataset


def get_dataloader(
    datasets: Union[str, List[str]],
    data_dir: str,
    include_reports: bool = False,
    include_labels: bool = False,
    cache_dataset: bool = False,
    cache_dir: Optional[str] = None,
    use_gds: bool = False,
    transform: Optional[Callable] = None,
    fraction: float = 1.0,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = True,
    persistent_workers: bool = True,
    use_thread: bool = False,
) -> data.DataLoader:
    """
    Get dataloader for training.
    """

    if isinstance(datasets, str):
        datasets = [datasets]

    # Validate constraints
    if include_reports:
        assert set(datasets).issubset({"ct_rate", "merlin", "inspect"}), \
            "When include_reports=True, only 'ct_rate', 'merlin', and 'inspect' are allowed."
    if include_labels:
        assert set(datasets).issubset({"abdomen_atlas", "abdomenct_1k"}), \
            "When include_labels=True, only 'abdomen_atlas' and 'abdomenct_1k' are allowed."
    if use_gds:
        assert cache_dataset, "GDS requires cache_dataset=True."
        assert torch.cuda.is_available(), "GDS requires CUDA to be available."

    # Dataset configurations
    DATASET_CONFIGS = {
        "ct_rate": {"folder": "CT-RATE", "base_name": "CTRate",
            "extra": {"include_reports": include_reports, "fraction": fraction}},
        "inspect": {"folder": "INSPECT", "base_name": "Inspect",
            "extra": {"include_reports": include_reports, "fraction": fraction}},
        "merlin": {"folder": "MERLIN", "base_name": "Merlin",
            "extra": {"include_reports": include_reports, "fraction": fraction}},
        "nlst": {"folder": "NLST", "base_name": "Nlst"},
        "amos": {"folder": "Amos", "base_name": "Amos"},
        "abdomen_atlas": {"folder": "AbdomenAtlas1.0Mini", "base_name": "AbdomenAtlas",
            "extra": {"include_labels": include_labels}},
        "panorama": {"folder": "PANORAMA", "base_name": "Panorama"},
        "abdomenct_1k": {"folder": "AbdomenCT-1K", "base_name": "AbdomenCT1K",
            "extra": {"include_labels": include_labels}},
    }
    
    datasets_list = []
    for ds in datasets:
        if ds.lower() not in DATASET_CONFIGS:
            raise NotImplementedError(f"Dataset {ds} not implemented.")

        cfg = DATASET_CONFIGS[ds.lower()]
        folder = cfg["folder"]
        extra_args = cfg.get("extra", {})

        kwargs = {
            "data_dir": os.path.join(data_dir, folder),
            "transform": transform,
            **extra_args,
        }

        base_name = cfg["base_name"]
        class_suffix = "Dataset"
        if cache_dataset:
            class_suffix = "GDSDataset" if use_gds else "PersistentDataset"

        class_name = f"{base_name}{class_suffix}"
        DatasetClass = getattr(__import__("spectre.data", fromlist=[class_name]), class_name)

        if cache_dataset:
            kwargs["cache_dir"] = os.path.join(cache_dir, folder)
            if use_gds:
                kwargs["device"] = torch.cuda.current_device()

        datasets_list.append(DatasetClass(**kwargs))

    dataset = datasets_list[0] if len(datasets_list) == 1 else ConcatDataset(datasets_list)

    loader_cls = getattr(data, "ThreadDataLoader" if use_thread else "DataLoader")
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "drop_last": drop_last,
    }

    if not use_thread:
        loader_kwargs.update({
            "pin_memory": pin_memory, 
            "persistent_workers": persistent_workers
        })
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn

    return loader_cls(**loader_kwargs)
