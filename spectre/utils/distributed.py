import os

import torch.distributed as dist


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
