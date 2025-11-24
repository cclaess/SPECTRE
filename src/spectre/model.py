import torch
import torch.nn as nn


MODEL_CONFIGS = {
    "spectre-small": {
        "name": "spectre-small",
        "backbone": "vit_small_patch16_128",
        "backbone_checkpoint_path_or_url": None,
        "backbone_kwargs": {},
        "feature_combiner": "feat_vit_small",
        "feature_combiner_checkpoint_path_or_url": None,
        "feature_combiner_kwargs": {},
        "description": "SPECTRE model with ViT-Small backbone and feature combiner.",
    },  # Pretrained/Distilled checkpoints will be added later
    "spectre-base": {
        "name": "spectre-base",
        "backbone": "vit_base_patch16_128",
        "backbone_checkpoint_path_or_url": None,
        "backbone_kwargs": {},
        "feature_combiner": "feat_vit_base",
        "feature_combiner_checkpoint_path_or_url": None,
        "feature_combiner_kwargs": {},
        "description": "SPECTRE model with ViT-Base backbone and feature combiner.",
    },  # Pretrained/Distilled checkpoints will be added later
    "spectre-large": {
        "name": "spectre-large",
        "backbone": "vit_large_patch16_128",
        "backbone_checkpoint_path_or_url": None,
        "backbone_kwargs": {},
        "feature_combiner": "feat_vit_large",
        "feature_combiner_checkpoint_path_or_url": None,
        "feature_combiner_kwargs": {},
        "description": "SPECTRE model with ViT-Large backbone and feature combiner.",
    },
    "spectre-large-pretrained": {
        "name": "spectre-large-pretrained",
        "backbone": "vit_large_patch16_128",
        "backbone_checkpoint_path_or_url": "https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_backbone_vit_large_patch16_128.pt?download=true",
        "backbone_kwargs": {
            "num_classes": 0,
            "global_pool": '',
            "pos_embed": "rope",
            "rope_kwargs": {"base": 1000.0},
            "init_values": 1.0,
        },
        "feature_combiner": "feat_vit_large",
        "feature_combiner_checkpoint_path_or_url": "https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_combiner_feature_vit_large.pt?download=true",
        "feature_combiner_kwargs": {
            "num_classes": 0,
            "global_pool": '',
            "pos_embed": "rope",
            "rope_kwargs": {"base": 100.0},
            "init_values": 1.0,
        },
        "description": "Pretrained SPECTRE model with ViT-Large backbone and feature combiner.",
    }
}


class SpectreImageFeatureExtractor(nn.Module):
    def __init__(
        self, 
        backbone_name: str, 
        backbone_kwargs: dict = {},
        backbone_checkpoint_path_or_url: str | None = None,
        feature_combiner_name: str | None = None, 
        feature_combiner_kwargs: dict = {},
        feature_combiner_checkpoint_path_or_url: str | None = None,
        **kwargs,
    ):  
        super().__init__()
        self.backbone = None
        self.feature_combiner = None
        self._init_backbone(
            backbone_name,
            checkpoint_path_or_url=backbone_checkpoint_path_or_url,
            **backbone_kwargs,
            **kwargs,
        )
        if feature_combiner_name is not None:
            self._init_feature_combiner(
                feature_combiner_name,
                checkpoint_path_or_url=feature_combiner_checkpoint_path_or_url,
                **feature_combiner_kwargs,
                **kwargs,
            )

    def _init_backbone(
        self, 
        model_name: str,
        checkpoint_path_or_url: str | None = None,
        **kwargs
    ):
        backbone_cls = getattr(__import__('spectre.models', fromlist=[model_name]), model_name)
        self.backbone = backbone_cls(
            checkpoint_path_or_url=checkpoint_path_or_url, 
            **kwargs,
        )
        
    def _init_feature_combiner(
        self, 
        model_name: str,
        checkpoint_path_or_url: str | None = None,
        **kwargs,
    ):
        if self.backbone.global_pool == '':
            patch_dim = self.backbone.embed_dim * 2  # CLS + AVG pooled tokens
        else:
            patch_dim = self.backbone.embed_dim

        feature_combiner_cls = getattr(__import__('spectre.models', fromlist=[model_name]), model_name)
        self.feature_combiner = feature_combiner_cls(
            patch_dim=patch_dim, 
            checkpoint_path_or_url=checkpoint_path_or_url, 
            **kwargs,
        )

    def extract_backbone_features(
        self, 
        x: torch.Tensor,
    ):
        """
        Extract features from the backbone for a batch of image sets. Input is expected to be of 
        shape (B, N, C, H, W, D), where B is the batch size, N is the number of image patches per 
        image, C is the number of channels, H is height, W is width, and D is depth.
        The output will be a tensor of extracted features (B, N, T, F) where T is the number of 
        tokens and F is the feature dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C, H, W, D)
        Returns:
            torch.Tensor: Extracted features of shape (B, N, T, F)
        """
        assert x.ndim == 6, "Input tensor must have 6 dimensions: (B, N, C, H, W, D)"
        B, N, C, H, W, D = x.shape
        x = x.view(B * N, C, H, W, D)
        features = self.backbone(x)
        if features.ndim == 2:  # only CLS token
            features = features.unsqueeze(1)
        features = features.view(B, N, features.shape[1], -1)
        return features

    def combine_features(
        self, 
        features: torch.Tensor,
        grid_size: tuple[int, int, int],
    ):
        """
        Combine features from multiple image patches using the feature combiner.

        Args:
            features (torch.Tensor): Input features of shape (B, N, T, F)
            grid_size (tuple[int, int, int]): Grid size of the image patches
        Returns:
            torch.Tensor: Combined features of shape (B, T', F')
        """
        _, N, T, _ = features.shape
        assert features.ndim == 4, "Input features must have 4 dimensions: (B, N, T, F)"
        assert N == grid_size[0] * grid_size[1] * grid_size[2], \
            "Number of patches N must match the product of grid_size dimensions"

        if T == 1:  # only CLS token
            features = features.squeeze(2)
        else:
            # We combine CLS tokens with AVG pooling of other tokens
            features = torch.cat([
                features[:, :, 0, :],  # CLS token (B, N, F)
                features[:, :, 1:, :].mean(dim=2)  # AVG pooled tokens (B, N, F)
            ], dim=-1)  # (B, N, 2F)
        features = self.feature_combiner(features, grid_size)  # (B, T', F')
        return features

    def forward(self, x, grid_size: tuple[int, int, int] | None = None):
        features = self.extract_backbone_features(x)
        if self.feature_combiner is not None:
            assert grid_size is not None, \
                "`grid_size` must be provided when using feature combiner"
            features = self.combine_features(features, grid_size)
        return features

    @classmethod
    def from_config(
        cls, 
        config: dict,
        **kwargs,
    ) -> 'SpectreImageFeatureExtractor':
        
        model = cls(
            backbone_name=config["backbone"],
            backbone_checkpoint_path_or_url=config.get("backbone_checkpoint_path_or_url", None),
            backbone_kwargs=config.get("backbone_kwargs", {}),
            feature_combiner_name=config.get("feature_combiner", None),
            feature_combiner_checkpoint_path_or_url=config.get("feature_combiner_checkpoint_path_or_url", None),
            feature_combiner_kwargs=config.get("feature_combiner_kwargs", {}),
            **kwargs,
        )
        return model
