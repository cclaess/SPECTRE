import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from spectre.model import SpectreImageFeatureExtractor
try:
    from .configuration_spectre import SpectreConfig
except ImportError:
    from configuration_spectre import SpectreConfig


class SpectreModel(PreTrainedModel):
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

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_size=None,
        return_dict=False,
        **kwargs,
    ):
        outputs = self.model(pixel_values, grid_size=grid_size)

        if not return_dict:
            return outputs

        return BaseModelOutput(last_hidden_state=outputs)
