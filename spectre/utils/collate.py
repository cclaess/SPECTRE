import random
from typing import List, Callable, Optional, Tuple

import torch
from monai.data import list_data_collate


def extended_collate_dino(
    samples_list: List, 
    mask_ratio: Optional[Tuple[float, float]] = None, 
    mask_probability: Optional[float] = None, 
    n_tokens: Optional[int] = None, 
    mask_generator: Optional[Callable] = None,
) -> dict:
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
    

def extended_collate_siglip(
    samples_list: List,
    tokenizer: Optional[Callable] = None,
    tokenizer_padding: bool = True,
    tokenizer_truncation: bool = True,
    tokenizer_max_length: int = 1024,
) -> dict:
    """
    Applies SigLIP collate and then extends it with tokenization logic.
    
    Args:
        samples_list: List of samples containing 'image' and 'report'.
        tokenizer: Tokenizer function to apply on the reports.
    
    Returns:
        A dictionary with collated images and tokenized text.
    """
    try:
        collated_data = list_data_collate(samples_list)
    except KeyError as e:
        elem = samples_list[0]
        data = [i for k in samples_list for i in k] if isinstance(elem, list) else samples_list
        keys = [d.keys() for d in data if isinstance(d, dict)]
        for k in keys:
            print(f"Keys in sample: {k}")
        raise e

    if "image" in collated_data.keys():
        if (
            hasattr(samples_list[0]["image"].data, "meta") 
            and "filename_or_obj" in samples_list[0]["image"].data.meta
        ):
            collated_data["filename"] = [s["image"].data.meta["filename_or_obj"] for s in samples_list]

    if tokenizer is not None and "report" in collated_data.keys():
        tokenizer_output = tokenizer.batch_encode_plus(
            collated_data["report"], 
            add_special_tokens=True,
            padding=tokenizer_padding,
            truncation=tokenizer_truncation,
            max_length=tokenizer_max_length,
        )
        
        collated_data["input_ids"] = torch.tensor(tokenizer_output["input_ids"])
        collated_data["attention_mask"] = torch.tensor(tokenizer_output["attention_mask"])

    return collated_data


def extract_patches_non_overlapping(x, patch_size=(128, 128, 64)):
    """
    x: (B, C, D, H, W)
    returns: (B, N, C, patch_d, patch_h, patch_w)
    where N = (D // patch_d) * (H // patch_h) * (W // patch_w)
    """
    B, C, H, W, D = x.shape
    patch_h, patch_w, patch_d = patch_size
    assert H % patch_h == 0 and W % patch_w == 0 and D % patch_d == 0, \
        "Volume must be divisible by patch size for this method."
    
    # Reshape into grid
    x = x.view(
        B, C,
        H // patch_h, patch_h,
        W // patch_w, patch_w,
        D // patch_d, patch_d,
    )
    
    # Rearrange so patches come together
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, H_blocks, W_blocks, D_blocks, C, patch_h, patch_w, patch_d)

    # Merge block indices into one dimension
    N = (H // patch_h) * (W // patch_w) * (D // patch_d)
    x = x.contiguous().view(B, N, C, patch_h, patch_w, patch_d)
    return x
