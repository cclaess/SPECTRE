import os

import torch.distributed as dist
from accelerate import Accelerator, DataLoaderConfiguration


def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_size() -> int:
    """
    Returns:
        Number of processes in the distributed group
    """
    if not is_enabled():
        return 1
    return dist.get_world_size()


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process in the distributed group
    """
    if not is_enabled():
        return 0
    return dist.get_rank()


def get_local_size() -> int:
    """
    Returns:
        Number of processes on the current machine
    """
    if not is_enabled():
        return 1
    return int(os.environ.get("LOCAL_SIZE", 1))


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process on the current machine
    """
    if not is_enabled():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def init_distributed(cfg):
    """
    Initialize distributed training.
    """

    # Initialize accelerator
    dataloader_config = DataLoaderConfiguration(
        non_blocking=cfg.train.pin_memory,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        log_with="wandb" if cfg.train.log_wandb else None,
        dataloader_config=dataloader_config,
    )

    # Initialize wandb
    if cfg.train.log_wandb:
        accelerator.init_trackers(
            project_name="spectre",
            config={k: v for d in cfg.values() for k, v in d.items()},
            init_kwargs={
                "dir": os.path.join(cfg.train.output_dir, "logs"),
            },
        )
    
    return accelerator
