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
    # images = []
    # reports = []
    # filenames = []

    # for sample in samples_list:
    #     filenames.append(sample[0]["image"].data.meta["filename_or_obj"])
    #     images.append(torch.stack([s["image"] for s in sample], dim=0))
    #     reports.append(sample[0]["report"])

    # collated_data = dict()
    # collated_data["filename"] = filenames
    # collated_data["image"] = torch.stack(images, dim=0)
    # collated_data["report"] = reports
    collated_data = list_data_collate(samples_list)
    if "image" in collated_data.keys():
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
