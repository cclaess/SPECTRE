import torch.nn as nn


def get_vit_lr_decay_rate(
    name: str,
    llrd_factor: float = 1.0,
    num_layers: int = 12,
    force_is_backbone: bool = False,
    shift: int = 0,
) -> float:
    """
    Get the layer-wise learning rate decay (LLRD) rate for a given parameter name.

    Args:
        name:
            The name of the parameter.
        llrd_factor:
            The decay factor for each layer.
        num_layers:
            The total number of layers in the model.
        force_is_backbone:
            If True, forces the function to treat the parameter as part of the backbone.
        shift:
            An integer to shift the layer ids, useful when combining multiple modules.

    Returns:
        The learning rate multiplier for the parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".patch_proj" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".reg_token" in name
        ):
            layer_id = 0
        elif ".blocks." in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1 + shift

    return llrd_factor ** (num_layers + 1 - layer_id)


def get_param_groups_with_decay(
    model: nn.Module,
    llrd_factor: float = 1.0,
    patch_embed_lr_mult: float = 1.0,
    projection_head_wd_mult: float = 1.0,
    lora_lr_factor: float = 1.0,
    num_layers: int | None = None,
):
    
    force_is_backbone = False
    shift = 0
    if num_layers is not None:
        num_layers = num_layers
    elif hasattr(model, "n_blocks"):
        num_layers = model.n_blocks
        force_is_backbone = True
    elif hasattr(model, "blocks"):
        num_layers = len(model.blocks)
        force_is_backbone = True
    elif hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        num_layers = len(model.backbone.blocks)
    elif hasattr(model, "backbone_student") and hasattr(model.backbone_student, "blocks"):  # DINO specific
        num_layers = len(model.backbone_student.blocks)
    elif hasattr(model, "backbone_student") and hasattr(model.backbone_student, "vit") and hasattr(model.backbone_student.vit, "blocks"):  # DINOv2 specific
        num_layers = len(model.backbone_student.vit.blocks)
    elif hasattr(model, "backbone_image") and hasattr(model.backbone_image, "blocks"):  # SigLIP specific
        if not hasattr(model, "feature_comb_image") or model.feature_comb_image is None:
            num_layers = len(model.backbone_image.blocks)
        else:
            num_layers = len(model.backbone_image.blocks) + len(model.feature_comb_image.blocks)
            shift = len(model.backbone_image.blocks)
            force_is_backbone = True
    else:
        num_layers = 0
    
    all_param_groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not "lora_" in n:
            s = shift if "feature_comb" in n else 0
            llrd_rate = get_vit_lr_decay_rate(
                n, llrd_factor, num_layers, force_is_backbone, s,
            )

            d = {
                "name": n,
                "params": p,
                "lr_mult": llrd_rate,
                "wd_mult": 1.0,
            }

            if "head" in n or "projection" in n:
                d["wd_mult"] = projection_head_wd_mult

            # No weight-decay on biases, norm parameters, layer scale gamma, learned tokens and embeddings
            if n.endswith("bias") or "norm" in n or "gamma" in n or "fourrier_w" in n:
                d["wd_mult"] = 0.0

            if "patch_embed" in n:
                d["lr_mult"] *= patch_embed_lr_mult
        
        else:
            # LoRA parameters
            d = {
                "name": n,
                "params": p,
                "lr_mult": lora_lr_factor,
                "wd_mult": 1.0,
            }
        
        all_param_groups.append(d)
    return all_param_groups
