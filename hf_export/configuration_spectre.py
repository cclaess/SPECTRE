from transformers import PretrainedConfig


class SpectreConfig(PretrainedConfig):
    model_type = "spectre"

    def __init__(
        self,
        backbone_name="vit_large_patch16_128",
        backbone_kwargs={
            "num_classes": 0,
            "global_pool": '',
            "pos_embed": "rope",
            "rope_kwargs": {"base": 1000.0},
            "init_values": 1.0,
        },
        feature_combiner_name="feat_vit_large",
        feature_combiner_kwargs={
            "num_classes": 0,
            "global_pool": "",
            "pos_embed": "rope",
            "rope_kwargs": {"base": 100.0},
            "init_values": 1.0,
        },
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_name = backbone_name
        self.backbone_kwargs = backbone_kwargs or {}
        self.feature_combiner_name = feature_combiner_name
        self.feature_combiner_kwargs = feature_combiner_kwargs or {}
