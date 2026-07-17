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
        features = model(scan).last_hidden_state

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
        **kwargs,
    ):
        """Embed one or more CT scans. See `SpectreImageFeatureExtractor.forward` for the shapes
        accepted; `grid_size` is required only for input that is already windowed into crops.
        """
        outputs = self.model(pixel_values, grid_size=grid_size, **kwargs)

        if not return_dict:
            return outputs

        return BaseModelOutput(last_hidden_state=outputs)
