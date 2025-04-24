import os
import random
import warnings
from typing import Optional

import torch
import numpy as np


def load_state(ckpt_path: str, **named_objects) -> int:
    """
    Loads state_dicts and epoch from a single checkpoint file into provided PyTorch objects.

    Parameters:
    - ckpt_path (str): Path to the checkpoint file (e.g., 'checkpoints/checkpoint.pt').
    - named_objects (kwargs): Any number of named PyTorch objects.

    Returns:
    - epoch (int): The stored epoch value, if any. If not found, returns 0.
    """
    if not os.path.isfile(ckpt_path):
        warnings.warn(f"Checkpoint file not found: {ckpt_path}")
        return 0

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    epoch = checkpoint.get('epoch', 0)

    for name, obj in named_objects.items():
        if name in checkpoint:
            obj.load_state_dict(checkpoint[name])
        else:
            warnings.warn(f"No state_dict found for '{name}' in checkpoint.")
    
    if "torch_random_state" in checkpoint:
        torch_random_state = checkpoint["torch_random_state"]
        torch.random.set_rng_state(torch_random_state)
    if "numpy_random_state" in checkpoint:
        numpy_random_state = checkpoint["numpy_random_state"]
        np.random.set_state(numpy_random_state)
    if "random_random_state" in checkpoint:
        random_random_state = checkpoint["random_random_state"]
        random.setstate(random_random_state)

    return epoch


def save_state(
    ckpt_path: str, 
    epoch: Optional[int] = None,
    torch_random_state: Optional[torch.Tensor] = None,
    numpy_random_state: Optional[tuple] = None,
    random_random_state: Optional[tuple] = None,
    **named_objects
):
    """
    Saves a single file containing the epoch and state_dicts of provided PyTorch objects.

    Parameters:
    - ckpt_path (str): Full path to the checkpoint file (e.g., 'checkpoints/checkpoint.pt').
    - epoch (int, optional): Epoch number or other metadata.
    - named_objects (kwargs): Any number of named PyTorch objects.
    """
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    checkpoint = {}
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if torch_random_state is not None:
        checkpoint["torch_random_state"] = torch_random_state
    if numpy_random_state is not None:
        checkpoint["numpy_random_state"] = numpy_random_state
    if random_random_state is not None:
        checkpoint["random_random_state"] = random_random_state
    for name, obj in named_objects.items():
        checkpoint[name] = obj.state_dict()

    torch.save(checkpoint, ckpt_path)
