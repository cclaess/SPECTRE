from typing import List

import torch
import torch.nn as nn


class SigLIPProjectionHead(nn.Module):
    """Projection head for SigLIP.

    Whereas SigLIP originally used a single linear layer for the projection
    head, we use a 3-layer MLP to deal with the partially frozen image and
    text backbones. This is similar to the DINO projection head without l2
    normalization (l2 normalization is performed in loss) and with LayerNorm.

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        output_dim:
            The output dimension of the head.
        layer_norm:
            Whether to use layer norm or not. Should be set to False when using
            a vision transformer backbone.
        freeze_last_layer:
            Number of epochs during which we keep the output layer fixed.
            Typically doing so during the first epoch helps training. Try
            increasing this value if the loss does not decrease.
        norm_last_layer:
            Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the
            training unstable.

    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 512,
        layer_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):  
        super().__init__()

        blocks = [
            (
                input_dim,
                hidden_dim,
                nn.LayerNorm(hidden_dim) if layer_norm else None,
                nn.GELU(),
            ),
            (
                hidden_dim,
                hidden_dim,
                nn.LayerNorm(hidden_dim) if layer_norm else None,
                nn.GELU(),
            ),
            (hidden_dim, bottleneck_dim, None, None),
        ]

        layers: List[nn.Module] = []
        for block in blocks:
            in_dim, out_dim, ln, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(ln)
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            if ln:
                layers.append(ln)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        # Tell mypy this is ok because fill_ is overloaded.
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        # Option to normalize last layer.
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel last layer gradients to stabilize the training."""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init._no_grad_trunc_normal_(
                module.weight,
                mean=0,
                std=0.02,
                a=-2,
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one forward pass through the head."""
        x = self.layers(x)
        x = self.last_layer(x)
        return x
