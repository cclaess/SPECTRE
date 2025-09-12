import os
import random
import warnings
from typing import Optional, Any

import torch
import numpy as np
import torch.distributed as dist


def _get_local_rng_state() -> dict:
    """Return a picklable dict with local RNG states (cpu & cuda, numpy, random)."""
    state = {
        "torch": torch.get_rng_state().cpu(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }

    if torch.cuda.is_available():
        # make sure CUDA states are stored on CPU so they are picklable
        cuda_states = [s.cpu() for s in torch.cuda.get_rng_state_all()]
        state["cuda"] = cuda_states
    else:
        state["cuda"] = None

    return state


def _set_local_rng_state(state: dict) -> None:
    """Set local RNG states from the dict produced by _get_local_rng_state()."""
    if state is None:
        return

    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and state["cuda"] is not None and torch.cuda.is_available():
        try:
            # move back to CUDA tensors for this process and set them
            cuda_states = [s.cuda() for s in state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            # fallback: try setting per-device RNG if set_rng_state_all fails
            for i, s in enumerate(state["cuda"]):
                try:
                    torch.cuda.set_rng_state(s.cuda(), device=i)
                except Exception:
                    # ignore if device mismatch
                    pass

    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "random" in state and state["random"] is not None:
        random.setstate(state["random"])


def save_state(ckpt_path: str, epoch: Optional[int] = None, **named_objects: Any) -> None:
    """
    Save a checkpoint that includes:
      - epoch (optional)
      - state_dicts for provided named_objects
      - rng_states: list of per-rank RNG dictionaries (one entry per world rank)

    If torch.distributed is initialized the RNG states from all ranks are gathered and
    stored in checkpoint["rng_states"] (list indexed by rank). Only rank 0 writes the file.
    In single-process mode the checkpoint contains a single-item rng_states list.
    """
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # prepare local RNG state
    local_rng = _get_local_rng_state()

    # distributed path: gather RNG states from all ranks
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        all_states = [None] * world_size
        # gather python objects (picklable)
        dist.all_gather_object(all_states, local_rng)

        # only rank 0 writes the checkpoint file
        if rank == 0:
            checkpoint = {}
            if epoch is not None:
                checkpoint["epoch"] = epoch
            checkpoint["rng_states"] = all_states

            # save provided objects' state_dicts (rank 0's local state_dicts)
            for name, obj in named_objects.items():
                checkpoint[name] = obj.state_dict()

            torch.save(checkpoint, ckpt_path)

        # ensure everyone waits until rank 0 finished writing
        dist.barrier()

    else:
        # single-process fallback
        checkpoint = {}
        if epoch is not None:
            checkpoint["epoch"] = epoch
        checkpoint["rng_states"] = [local_rng]
        for name, obj in named_objects.items():
            checkpoint[name] = obj.state_dict()
        torch.save(checkpoint, ckpt_path)


def load_state(ckpt_path: str, **named_objects: Any) -> int:
    """
    Load checkpoint saved by save_state.

    - Each process loads the same file and restores its own RNG state (checkpoint['rng_states'][rank]).
    - Named objects that exist in the checkpoint will have their state_dict loaded.
    - Returns epoch (int) if present, otherwise 0.
    """
    if not os.path.isfile(ckpt_path):
        warnings.warn(f"Checkpoint file not found: {ckpt_path}")
        return 0

    # load on all ranks (shared FS assumed)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", 0)

    # load state_dicts into provided objects
    for name, obj in named_objects.items():
        if name in checkpoint:
            try:
                obj.load_state_dict(checkpoint[name])
            except Exception as e:
                warnings.warn(f"Failed to load state_dict for '{name}': {e}")
        else:
            warnings.warn(f"No state_dict found for '{name}' in checkpoint.")

    # restore this rank's RNG state
    rng_states = checkpoint.get("rng_states", None)
    if rng_states is not None:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank < len(rng_states):
                my_state = rng_states[rank]
            else:
                my_state = None
        else:
            # single-process file: first element
            my_state = rng_states[0] if len(rng_states) > 0 else None

        try:
            _set_local_rng_state(my_state)
        except Exception as e:
            warnings.warn(f"Failed to restore RNG state: {e}")

    else:
        warnings.warn("No 'rng_states' found in checkpoint; RNGs not restored.")

    return epoch


def extract_model_from_checkpoint_dinov2(checkpoint_path: str):
    # Load the checkpoint
    checkpoint = torch.load(
        checkpoint_path, 
        weights_only=False, 
        map_location="cpu"
    )
    
    # Get model state dict
    model_state = checkpoint.get("model", checkpoint)

    # Create output folder
    output_dir = str(checkpoint_path).replace(".pt", "")
    os.makedirs(output_dir, exist_ok=True)

    # Quick check: compare the parameters of head_teacher_ibot vs head_teacher_dino
    teacher_dino_keys = [k for k in model_state.keys() if k.startswith("head_teacher_dino.")]
    teacher_ibot_keys = [k for k in model_state.keys() if k.startswith("head_teacher_ibot.")]

    ibot_separate = True
    if teacher_dino_keys and teacher_ibot_keys:
        if all(torch.equal(model_state[dino_key], model_state[ibot_key]) \
               for dino_key, ibot_key in zip(teacher_dino_keys, teacher_ibot_keys)):
            ibot_separate = False  # Same weights → no separate ibot head
    
    # Define the components to save
    components = {
        "backbone_teacher.pt": "backbone_teacher.vit",
        "backbone_student.pt": "backbone_student.vit",
        "head_student_dino.pt": "head_student_dino",
        "head_teacher_dino.pt": "head_teacher_dino"
    }

    # Add ibot heads only if separate
    if ibot_separate:
        components["head_student_ibot.pt"] = "head_student_ibot"
        components["head_teacher_ibot.pt"] = "head_teacher_ibot"

    # Extract and save each component
    for filename, key in components.items():
        sub_state_dict = {k.replace(f"{key}.", ""): v for k, v in model_state.items() if k.startswith(key)}
        if not sub_state_dict:
            print(f"[WARNING] No parameters found for {key}, skipping...")
            continue
        torch.save(sub_state_dict, os.path.join(output_dir, filename))
    
    print(f"Components extracted to: {output_dir}")
