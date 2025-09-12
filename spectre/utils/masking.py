import math
from typing import Optional, Tuple, Union

import torch


def _random_block_mask(
    size: Tuple[int, int, int],
    num_masks: int,
    min_num_masks_per_block: int = 4,
    max_num_masks_per_block: Optional[int] = None,
    max_attempts_per_block: int = 10,
    generator: Optional[torch.Generator] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """3D helper: generate a (H, W, D) boolean mask by placing cuboidal blocks.

    - size: (H, W, D)
    - num_masks: target total number of masked voxels for this image
    - min_num_masks_per_block / max_num_masks_per_block: voxel-range per block
    """
    H, W, D = size
    total = H * W * D
    num_masks = min(max(0, int(num_masks)), total)

    if max_num_masks_per_block is None:
        max_num_masks_per_block = max(1, num_masks)

    mask = torch.zeros((H, W, D), dtype=torch.bool, device=device)
    masked_count = 0
    global_attempts = 0

    orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    # Try to place blocks until we have enough masked voxels or we exceed attempts
    while masked_count < num_masks and global_attempts < max_attempts_per_block:
        global_attempts += 1

        # choose target voxels for this block
        target_voxels = int(torch.randint(
            min_num_masks_per_block, max_num_masks_per_block + 1, (1,), generator=generator
        ).item())

        found = False
        local_attempts = 0
        while not found and local_attempts < max_attempts_per_block:
            local_attempts += 1

            # random pick order for dims to reduce bias
            order_idx = int(torch.randint(0, 3, (1,), generator=generator).item())
            order = orders[order_idx]

            # pick first dimension
            if order[0] == 0:
                h = int(torch.randint(1, min(H, target_voxels) + 1, (1,), generator=generator).item())
            elif order[0] == 1:
                w = int(torch.randint(1, min(W, target_voxels) + 1, (1,), generator=generator).item())
            else:
                d = int(torch.randint(1, min(D, target_voxels) + 1, (1,), generator=generator).item())

            # progressively choose remaining dims while ensuring feasibility
            try:
                if order[0] == 0:
                    # h chosen -> pick w then compute d_needed
                    max_w = max(1, min(W, target_voxels // h))
                    w = int(torch.randint(1, max_w + 1, (1,), generator=generator).item())
                    d_needed = math.ceil(target_voxels / (h * w))
                    if d_needed <= D:
                        d = max(1, d_needed)
                        found = True
                elif order[0] == 1:
                    # w chosen -> pick d then compute h_needed
                    max_d = max(1, min(D, target_voxels // w))
                    d = int(torch.randint(1, max_d + 1, (1,), generator=generator).item())
                    h_needed = math.ceil(target_voxels / (d * w))
                    if h_needed <= H:
                        h = max(1, h_needed)
                        found = True
                else:
                    # d chosen -> pick h then compute w_needed
                    max_h = max(1, min(H, target_voxels // d))
                    h = int(torch.randint(1, max_h + 1, (1,), generator=generator).item())
                    w_needed = math.ceil(target_voxels / (d * h))
                    if w_needed <= W:
                        w = max(1, w_needed)
                        found = True

            except ValueError:
                # in case of invalid ranges (defensive); just continue trying
                continue

            # fallback alternative attempt: try simple factorization heuristics
            if not found:
                # attempt small-to-large factorization
                for hh in range(1, min(H, target_voxels) + 1):
                    for ww in range(1, min(W, target_voxels // hh) + 1):
                        dd = math.ceil(target_voxels / (hh * ww))
                        if dd <= D:
                            h, w, d = hh, ww, dd
                            found = True
                            break
                    if found:
                        break

        if not found:
            # couldn't find a fitting block this global attempt; move on
            continue

        # clamp block dims to volume just in case and ensure at least 1
        h = min(max(1, int(h)), H)
        w = min(max(1, int(w)), W)
        d = min(max(1, int(d)), D)

        # choose random location so block fits
        x0 = int(torch.randint(0, (H - h) + 1, (1,), generator=generator).item()) if H - h > 0 else 0
        y0 = int(torch.randint(0, (W - w) + 1, (1,), generator=generator).item()) if W - w > 0 else 0
        z0 = int(torch.randint(0, (D - d) + 1, (1,), generator=generator).item()) if D - d > 0 else 0

        mask[x0 : x0 + h, y0 : y0 + w, z0 : z0 + d] = True
        masked_count = int(mask.sum().item())

    # If still short, fill remaining voxels at random positions
    if masked_count < num_masks:
        remaining = num_masks - masked_count
        indices = torch.nonzero(~mask, as_tuple=False)
        if indices.numel() > 0:
            perm = torch.randperm(indices.shape[0], generator=generator, device=mask.device)
            pick = indices[perm[:remaining]]
            mask[pick[:, 0], pick[:, 1], pick[:, 2]] = True

    return mask


def random_block_mask(
    size: Tuple[int, int, int, int],
    batch_mask_ratio: float = 0.5,
    min_image_mask_ratio: float = 0.1,
    max_image_mask_ratio: float = 0.5,
    min_num_masks_per_block: int = 4,
    max_num_masks_per_block: Optional[int] = None,
    max_attempts_per_block: int = 10,
    generator: Optional[torch.Generator] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Create random block masks for 3D volumes only.

    Args:
        size: (B, H, W, D)
        batch_mask_ratio: fraction of images in the batch to apply masking to
        min_image_mask_ratio / max_image_mask_ratio: per-image mask fraction range
        min_num_masks_per_block / max_num_masks_per_block: voxels per block range
        max_attempts_per_block: attempts to find a fitting block
        generator: optional torch.Generator for reproducibility.
        device: device for returned tensor

    Returns:
        boolean tensor with shape (B, H, W, D)
    """
    if len(size) != 4:
        raise ValueError("size must be (B, H, W, D) for 3D masking.")

    B, H, W, D = size

    if max_image_mask_ratio < min_image_mask_ratio:
        raise ValueError("max_image_mask_ratio must be >= min_image_mask_ratio.")

    num_images_masked = int(B * batch_mask_ratio)
    probs = torch.linspace(min_image_mask_ratio, max_image_mask_ratio, num_images_masked + 1).tolist()

    image_masks = []
    total_voxels = H * W * D

    for prob_min, prob_max in zip(probs[:-1], probs[1:]):
        # choose number of masked voxels for this image
        u = float(prob_min + (prob_max - prob_min) * torch.rand(1, generator=generator).item())
        num_mask = int(total_voxels * u)
        image_masks.append(
            _random_block_mask(
                size=(H, W, D),
                num_masks=num_mask,
                min_num_masks_per_block=min_num_masks_per_block,
                max_num_masks_per_block=max_num_masks_per_block,
                max_attempts_per_block=max_attempts_per_block,
                generator=generator,
                device=device,
            )
        )

    # Add non-masked images (all False) to fill the batch
    for _ in range(num_images_masked, B):
        image_masks.append(torch.zeros((H, W, D), dtype=torch.bool, device=device))

    perm = torch.randperm(B, generator=generator).tolist()
    image_masks = [image_masks[i] for i in perm]
    
    return torch.stack(image_masks)
