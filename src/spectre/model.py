import copy
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from spectre.presets import (
    BACKBONE_KWARGS,
    COMBINER_KWARGS,
    get_preset,
    list_presets,
    list_pretrained,
)
from spectre.windowing import window_scan


def _legacy_model_configs() -> Dict[str, Dict[str, Any]]:
    """The pre-0.2 `MODEL_CONFIGS` payload.

    The architecture kwargs come from `spectre.presets`, but the *layout* is reproduced by hand
    because the two registries disagree about what `"spectre-large"` means: here it is a
    weightless architecture and `"spectre-large-pretrained"` carries the weights, whereas
    `from_pretrained("spectre-large")` loads weights. Deriving the mapping would silently change
    which checkpoint existing callers get. `tests/test_legacy_api.py` pins the result.
    """
    def entry(name, backbone, combiner, backbone_ckpt, combiner_ckpt, description):
        return {
            "name": name,
            "backbone": backbone,
            "backbone_checkpoint_path_or_url": backbone_ckpt,
            "backbone_kwargs": copy.deepcopy(BACKBONE_KWARGS),
            "feature_combiner": combiner,
            "feature_combiner_checkpoint_path_or_url": combiner_ckpt,
            "feature_combiner_kwargs": copy.deepcopy(COMBINER_KWARGS),
            "description": description,
        }

    return {
        "spectre-small": entry(
            "spectre-small", "vit_small_patch16_128", "feat_vit_small", None, None,
            "SPECTRE model with ViT-Small backbone and feature combiner.",
        ),
        "spectre-base": entry(
            "spectre-base", "vit_base_patch16_128", "feat_vit_base", None, None,
            "SPECTRE model with ViT-Base backbone and feature combiner.",
        ),
        "spectre-large": entry(
            "spectre-large", "vit_large_patch16_128", "feat_vit_large", None, None,
            "SPECTRE model with ViT-Large backbone and feature combiner.",
        ),
        "spectre-large-pretrained": entry(
            "spectre-large-pretrained", "vit_large_patch16_128", "feat_vit_large",
            "https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_backbone_vit_large_patch16_128.pt?download=true",
            "https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_combiner_feature_vit_large.pt?download=true",
            "Pretrained SPECTRE model with ViT-Large backbone and feature combiner.",
        ),
    }


_MODEL_CONFIGS_DEPRECATION = (
    "spectre.MODEL_CONFIGS is deprecated since spectre-fm 0.2.1 and will be removed in 0.3.0. "
    "Use SpectreImageFeatureExtractor.from_pretrained('spectre-large') for the pretrained model, "
    "or spectre.list_presets() to see the available architectures."
)


class _DeprecatedModelConfigs(dict):
    """`MODEL_CONFIGS` that warns on access.

    A dict subclass rather than a module-level `__getattr__`, because `spectre/__init__.py`
    imports the name eagerly - a module hook would fire on every `import spectre`, including
    inside the Hub's trust_remote_code import.
    """

    def __getitem__(self, key):
        warnings.warn(_MODEL_CONFIGS_DEPRECATION, FutureWarning, stacklevel=2)
        return super().__getitem__(key)

    def get(self, key, default=None):
        # dict.get does not route through __getitem__ in CPython, so it needs its own warning.
        warnings.warn(_MODEL_CONFIGS_DEPRECATION, FutureWarning, stacklevel=2)
        return super().get(key, default)


MODEL_CONFIGS = _DeprecatedModelConfigs(_legacy_model_configs())


class SpectreImageFeatureExtractor(nn.Module):
    """SPECTRE image encoder: a ViT backbone over CT crops, plus an optional feature combiner.

    The backbone embeds each `crop_size` crop independently; the feature combiner then attends
    over the grid of crop embeddings to produce one embedding for the whole scan.

    The usual way in is `from_pretrained`:

        model = SpectreImageFeatureExtractor.from_pretrained("spectre-large")
        features = model(scan)                       # scan: (C, H, W, D) in Hounsfield Units
        features = model([scan_a, scan_b])           # differing sizes, one backbone pass

    Pass `include_feature_combiner=False` for per-crop backbone features only.
    """

    def __init__(
        self,
        backbone_name: str,
        backbone_kwargs: Optional[dict] = None,
        backbone_checkpoint_path_or_url: str | None = None,
        feature_combiner_name: str | None = None,
        feature_combiner_kwargs: Optional[dict] = None,
        feature_combiner_checkpoint_path_or_url: str | None = None,
        **kwargs,
    ):
        # This signature is a published contract: the Hugging Face wrapper
        # (hf_export/modeling_spectre.py) calls it with exactly these keywords.
        super().__init__()
        self.backbone = None
        self.feature_combiner = None
        self._init_backbone(
            backbone_name,
            checkpoint_path_or_url=backbone_checkpoint_path_or_url,
            **(backbone_kwargs or {}),
            **kwargs,
        )
        if feature_combiner_name is not None:
            self._init_feature_combiner(
                feature_combiner_name,
                checkpoint_path_or_url=feature_combiner_checkpoint_path_or_url,
                **(feature_combiner_kwargs or {}),
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

    @classmethod
    def from_pretrained(
        cls,
        name: str = "spectre-large",
        *,
        include_feature_combiner: bool = True,
        pretrained: bool = True,
        strict: bool = True,
        backbone_kwargs: Optional[dict] = None,
        feature_combiner_kwargs: Optional[dict] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
    ) -> 'SpectreImageFeatureExtractor':
        """Build a named SPECTRE model, by default with its published weights.

        Args:
            name: A preset name, e.g. "spectre-large". See `spectre.list_presets()`.
            include_feature_combiner: Include the feature combiner (one embedding per scan). When
                False you get per-crop backbone features and no combiner checkpoint is fetched.
            pretrained: Load published weights. Raises if the preset has none.
            strict: Require the checkpoint to match the architecture exactly.
            backbone_kwargs: Overrides merged over the preset's backbone kwargs.
            feature_combiner_kwargs: Overrides merged over the preset's combiner kwargs.
            device: Device to move the model to.
            dtype: Dtype to cast the model to.
            verbose: Print checkpoint download/load progress.

        Returns:
            The model in eval mode.
        """
        if any(token in str(name) for token in ('/', '\\', '://')) or str(name).endswith('.pt'):
            raise ValueError(
                f"from_pretrained() takes a preset name, not a path or URL, but got {name!r}. "
                f"Available presets: {list_presets()}. To load your own checkpoint, construct "
                f"the model directly:\n"
                f"    SpectreImageFeatureExtractor(\n"
                f"        backbone_name='vit_large_patch16_128',\n"
                f"        backbone_checkpoint_path_or_url={name!r},\n"
                f"    )"
            )

        preset = get_preset(name)

        if pretrained and not preset.has_pretrained_weights:
            raise ValueError(
                f"Preset {name!r} has no published weights. Presets with weights: "
                f"{list_pretrained()}. Pass pretrained=False for a randomly initialised "
                f"{name!r} model."
            )

        merged_backbone_kwargs = dict(preset.backbone_kwargs)
        merged_backbone_kwargs.update(backbone_kwargs or {})
        merged_combiner_kwargs = dict(preset.feature_combiner_kwargs)
        merged_combiner_kwargs.update(feature_combiner_kwargs or {})

        # __init__ forwards **kwargs into both sub-model factories, which only accept verbose and
        # strict when they are actually loading a checkpoint.
        load_kwargs = {"verbose": verbose, "strict": strict} if pretrained else {}

        model = cls(
            backbone_name=preset.backbone,
            backbone_kwargs=merged_backbone_kwargs,
            backbone_checkpoint_path_or_url=preset.backbone_weights if pretrained else None,
            feature_combiner_name=preset.feature_combiner if include_feature_combiner else None,
            feature_combiner_kwargs=merged_combiner_kwargs,
            feature_combiner_checkpoint_path_or_url=(
                preset.feature_combiner_weights if pretrained and include_feature_combiner else None
            ),
            **load_kwargs,
        )

        if device is not None or dtype is not None:
            model = model.to(device=device, dtype=dtype)
        return model.eval()

    @property
    def crop_size(self) -> Tuple[int, int, int]:
        """Spatial size (H, W, D) of one CT crop the backbone consumes."""
        return tuple(self.backbone.patch_embed.img_size)

    @property
    def has_feature_combiner(self) -> bool:
        return self.feature_combiner is not None

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

    def _pool_crop_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """Reduce per-crop backbone tokens (M, T, F) to the combiner's input (M, 2F).

        Must stay bit-identical to `combine_features`: the same CLS token concatenated with the
        same mean over the same axis. The prefix index 0 assumes num_prefix_tokens == 1, which
        holds for every preset; changing it would move published embeddings.
        """
        if features.ndim == 2:  # backbone already pooled globally
            return features
        if features.shape[1] == 1:  # CLS only
            return features[:, 0]
        return torch.cat([features[:, 0], features[:, 1:].mean(dim=1)], dim=-1)

    def _encode(
        self,
        items: List[Tuple[torch.Tensor, Optional[Tuple[int, int, int]]]],
        max_crops_per_forward: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Run crops from any number of scans through the backbone in one pass.

        `items` pairs each scan's crops (N_i, C, cH, cW, cD) with its grid. Crops from every scan
        go through the backbone together; only the feature combiner, whose positional encoding
        depends on the grid, is split back out per scan.
        """
        if not items:
            return []

        crops_per_scan = [crops for crops, _ in items]
        grids = [grid for _, grid in items]

        reference = crops_per_scan[0].shape[1:]
        for i, crops in enumerate(crops_per_scan):
            if crops.shape[1:] != reference:
                raise ValueError(
                    f"All scans must have the same crop shape, but scan 0 has crops of "
                    f"{tuple(reference)} and scan {i} has {tuple(crops.shape[1:])}."
                )

        param = next(self.parameters())
        counts = [int(crops.shape[0]) for crops in crops_per_scan]
        all_crops = torch.cat(
            [crops.to(device=param.device, dtype=param.dtype) for crops in crops_per_scan], dim=0
        )

        chunk = max_crops_per_forward if max_crops_per_forward else all_crops.shape[0]
        chunk_features = []
        for crops in torch.split(all_crops, chunk, dim=0):
            features = self.backbone(crops)
            if features.ndim == 2:  # only CLS token
                features = features.unsqueeze(1)
            if self.feature_combiner is not None:
                # Pool inside the loop: unpooled tokens for a whole batch of scans are orders of
                # magnitude larger than the pooled features the combiner actually needs.
                features = self._pool_crop_tokens(features)
            chunk_features.append(features)

        features = torch.cat(chunk_features, dim=0)
        per_scan = torch.split(features, counts, dim=0)

        if self.feature_combiner is None:
            return list(per_scan)

        # The combiner's rotary embedding depends on the grid and there is no attention mask, so
        # scans can only share a forward pass when their grids match exactly.
        groups: Dict[Tuple[int, int, int], List[int]] = {}
        for index, grid in enumerate(grids):
            groups.setdefault(tuple(grid), []).append(index)

        results: List[torch.Tensor] = [torch.empty(0)] * len(items)
        for grid, indices in groups.items():
            batch = torch.stack([per_scan[i] for i in indices], dim=0)  # (G, N, 2F)
            combined = self.feature_combiner(batch, grid)  # (G, T', F')
            for position, index in enumerate(indices):
                results[index] = combined[position]

        return results

    def _prepare_scan(
        self,
        x: torch.Tensor,
        grid_size: Optional[Sequence[int]],
        index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Normalise one scan to (crops, grid), windowing it first if it is still a whole volume."""
        where = "" if index is None else f" (scan {index})"

        if hasattr(x, 'as_tensor'):  # MONAI MetaTensor, without importing monai
            x = x.as_tensor()
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor{where}, got {type(x).__name__}.")

        crop_size = self.crop_size

        if x.ndim in (3, 4):
            if grid_size is not None:
                raise ValueError(
                    f"grid_size must not be passed for a raw {x.ndim}D scan{where}: it is derived "
                    f"from the scan's shape while windowing. Pass grid_size only for "
                    f"pre-windowed (N, C, H, W, D) input."
                )
            expected_channels = self.backbone.patch_embed.proj.in_channels
            if x.ndim == 4 and x.shape[0] != expected_channels:
                raise ValueError(
                    f"A 4D input{where} is read as one raw scan (C, H, W, D) with C="
                    f"{expected_channels}, but got shape {tuple(x.shape)}. For a batch of scans, "
                    f"pass a list of tensors instead."
                )
            return window_scan(x, crop_size)

        if x.ndim == 5:
            # (N, C, H, W, D) and (B, C, H, W, D) are indistinguishable by rank, so 5D always
            # means one pre-windowed scan; raw batches go through a list.
            if tuple(x.shape[2:]) != crop_size:
                raise ValueError(
                    f"A 5D input{where} is read as one pre-windowed scan (N, C, H, W, D) whose "
                    f"crops must be {crop_size}, but got crops of {tuple(x.shape[2:])}. If this "
                    f"is a batch of raw scans, pass a list of (C, H, W, D) tensors instead."
                )
            if grid_size is None:
                raise ValueError(
                    f"grid_size is required for pre-windowed (N, C, H, W, D) input{where}. Pass "
                    f"the raw scan as (C, H, W, D) to have it windowed for you."
                )
            grid = tuple(int(g) for g in grid_size)
            if len(grid) != 3:
                raise ValueError(f"grid_size must have 3 elements (H, W, D), got {grid}.")
            expected = grid[0] * grid[1] * grid[2]
            if x.shape[0] != expected:
                raise ValueError(
                    f"grid_size {grid} implies {expected} crops but got {x.shape[0]}{where}."
                )
            return x, grid

        raise ValueError(
            f"Unsupported input with {x.ndim} dimensions{where}. Expected one of: (H, W, D) or "
            f"(C, H, W, D) for a raw scan, (N, C, H, W, D) for a pre-windowed scan, "
            f"(B, N, C, H, W, D) for a pre-windowed batch, or a list of scans."
        )

    def forward(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        grid_size: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        *,
        max_crops_per_forward: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Embed one or more CT scans.

        Accepts, dispatching on rank:

        | input                | meaning                        | grid_size | returns          |
        |----------------------|--------------------------------|-----------|------------------|
        | `(H, W, D)`          | raw scan, in HU                | derived   | `(T', F')`       |
        | `(C, H, W, D)`       | raw scan, in HU                | derived   | `(T', F')`       |
        | `(N, C, cH, cW, cD)` | pre-windowed scan              | required  | `(T', F')`       |
        | `(B, N, C, ...)`     | pre-windowed batch, one grid   | required  | `(B, T', F')`    |
        | `list[...]`          | batch of scans, may differ     | per scan  | `list[(T', F')]` |

        Raw input is HU-windowed and cut into crops internally; pre-windowed input is passed
        through untouched, so it must already be preprocessed the same way.

        Without a feature combiner the trailing shape is the backbone's per-crop tokens
        `(N, T, F)` instead of `(T', F')`.

        Args:
            x: Scan(s), as above.
            grid_size: Crop grid for pre-windowed input. For a list, either one grid shared by
                every scan or one per scan.
            max_crops_per_forward: Cap the crops in a single backbone pass to bound memory.
                None runs them all at once.
        """
        is_batch = isinstance(x, (list, tuple))

        if is_batch:
            grids = self._grids_for_batch(grid_size, len(x))
            items = [self._prepare_scan(scan, g, i) for i, (scan, g) in enumerate(zip(x, grids))]
            return self._encode(items, max_crops_per_forward)

        if hasattr(x, 'as_tensor'):
            x = x.as_tensor()

        if isinstance(x, torch.Tensor) and x.ndim == 6:
            if grid_size is None:
                raise ValueError(
                    "grid_size is required for pre-windowed (B, N, C, H, W, D) input. Pass a list "
                    "of raw (C, H, W, D) scans to have them windowed for you."
                )
            grid = tuple(int(g) for g in grid_size)
            items = [self._prepare_scan(scan, grid, i) for i, scan in enumerate(x)]
            return torch.stack(self._encode(items, max_crops_per_forward), dim=0)

        items = [self._prepare_scan(x, grid_size)]
        return self._encode(items, max_crops_per_forward)[0]

    @staticmethod
    def _grids_for_batch(
        grid_size: Optional[Union[Sequence[int], Sequence[Sequence[int]]]],
        count: int,
    ) -> List[Optional[Tuple[int, ...]]]:
        """Resolve `grid_size` for a list input: one shared grid, one per scan, or none."""
        if grid_size is None:
            return [None] * count
        # A grid is always three ints, so all-ints means one shared grid.
        if all(isinstance(g, int) for g in grid_size):
            return [tuple(int(g) for g in grid_size)] * count
        grids = [tuple(int(v) for v in g) for g in grid_size]
        if len(grids) != count:
            raise ValueError(
                f"Got {len(grids)} grid sizes for {count} scans. Pass one grid_size per scan, or "
                f"a single (H, W, D) grid shared by all of them."
            )
        return grids

    @torch.inference_mode()
    def extract(
        self,
        scans: Union[torch.Tensor, Sequence[torch.Tensor]],
        grid_size: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        *,
        max_crops_per_forward: Optional[int] = 16,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """`forward` without gradients, with memory capped by default. See `forward`."""
        return self.forward(scans, grid_size, max_crops_per_forward=max_crops_per_forward)

    @classmethod
    def from_config(
        cls,
        config: dict,
        **kwargs,
    ) -> 'SpectreImageFeatureExtractor':
        warnings.warn(
            "SpectreImageFeatureExtractor.from_config() is deprecated since spectre-fm 0.2.1 and "
            "will be removed in 0.3.0. Use "
            "SpectreImageFeatureExtractor.from_pretrained('spectre-large') instead, or call "
            "SpectreImageFeatureExtractor(...) directly for a custom architecture.",
            FutureWarning,
            stacklevel=2,
        )
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
