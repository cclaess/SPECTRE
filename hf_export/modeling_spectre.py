import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from spectre.model import SpectreImageFeatureExtractor

try:
    from .configuration_spectre import SpectreConfig
except ImportError:
    from configuration_spectre import SpectreConfig


class SpectreModel(PreTrainedModel):
    """`transformers` wrapper around `SpectreImageFeatureExtractor`.

        from transformers import AutoModel
        model = AutoModel.from_pretrained("cclaess/SPECTRE-Large", trust_remote_code=True)

        # A raw CT scan (C, H, W, D) in Hounsfield Units - windowed internally.
        features = model(scan)                       # -> Tensor
        features = model(scan, return_dict=True).last_hidden_state

    Note `return_dict` defaults to False here rather than following `config.use_return_dict`;
    that is the behaviour the published model has always had, so it stays.

    This wrapper takes a single tensor, because `BaseModelOutput` holds a tensor. For batches of
    differently-sized scans - where all crops share one backbone pass - use the underlying
    extractor directly, which accepts a list:

        features = model.model.extract([scan_a, scan_b])   # -> list of (T', F')
    """

    config_class = SpectreConfig
    base_model_prefix = "spectre"
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)

        self.model = SpectreImageFeatureExtractor(
            backbone_name=config.backbone_name,
            backbone_kwargs=config.backbone_kwargs,
            feature_combiner_name=config.feature_combiner_name,
            feature_combiner_kwargs=config.feature_combiner_kwargs,
        )

        self.post_init()

    @property
    def crop_size(self):
        """Spatial size (H, W, D) of one CT crop the backbone consumes."""
        return self.model.crop_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_size=None,
        return_dict=False,
        max_crops_per_forward=None,
        output_hidden_states=None,
        output_attentions=None,
        **kwargs,
    ):
        """Embed one or more CT scans.

        See `SpectreImageFeatureExtractor.forward` for the shapes accepted; `grid_size` is
        required only for input that is already windowed into crops, and
        `max_crops_per_forward` caps how many crops go through the backbone at once.

        Args are named explicitly rather than splatted into the extractor: `**kwargs` here is a
        sink for the kwargs `transformers` callers pass out of habit (`use_cache` and friends),
        which are meaningless for a feature extractor. Forwarding them would turn a harmless
        call into a TypeError.
        """
        if output_hidden_states:
            raise NotImplementedError(
                "SPECTRE does not expose intermediate hidden states through this wrapper. It "
                "embeds each crop with a backbone and then attends over the grid of crop "
                "embeddings, so there is no single stack of layers whose outputs share a shape. "
                "For per-crop backbone layers, call "
                "model.model.backbone.forward_intermediates(crops) directly."
            )
        if output_attentions:
            raise NotImplementedError(
                "SPECTRE does not expose attention weights. Its attention uses a fused kernel "
                "(torch.nn.functional.scaled_dot_product_attention), which does not return them."
            )

        outputs = self.model(
            pixel_values,
            grid_size=grid_size,
            max_crops_per_forward=max_crops_per_forward,
        )

        if not return_dict:
            return outputs

        return BaseModelOutput(last_hidden_state=outputs)
