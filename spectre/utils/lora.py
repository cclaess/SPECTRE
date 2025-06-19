import torch.nn as nn
import loralib as lora


def add_lora_adapters(
        root_module: nn.Module,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_keywords: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    ) -> None:
    """
    Recursively traverses the model and replaces every `nn.Linear`
    whose name contains one of `target_keywords` with a LoRA-augmented
    linear layer from loralib.
    """

    for name, child in list(root_module.named_children()):
        # If the child is itself a container, recurse first
        add_lora_adapters(child, r, lora_alpha, lora_dropout, target_keywords)

        # Replace target linear layers
        if isinstance(child, nn.Linear) and any(k in name for k in target_keywords):
            lora_layer = lora.Linear(                             # loralib wrapper
                in_features=child.in_features,
                out_features=child.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=child.bias is not None,
            )

            # copy original weights so that behaviour is identical pre-training
            lora_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                lora_layer.bias.data = child.bias.data.clone()

            setattr(root_module, name, lora_layer)               # hot-swap!
