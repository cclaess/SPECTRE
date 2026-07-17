from transformers import PretrainedConfig

from spectre.presets import get_preset
from spectre.windowing import DEFAULT_CROP_SIZE, DEFAULT_HU_RANGE


class SpectreConfig(PretrainedConfig):
    """Configuration of a pretrained SPECTRE model.

    Nothing about the model is written down twice. The architecture is resolved from a named
    preset in `spectre.presets`, and the preprocessing defaults come from `spectre.windowing` -
    the same values the package itself uses. The resolved architecture is still serialised into
    `config.json`, so the published config stays self-describing.

    The fields above the architecture are the input contract: SPECTRE consumes a grid of
    `crop_size` crops cut from a CT scan oriented to `orientation` and windowed from `hu_range`
    onto [0, 1] at `voxel_spacing` (None meaning the scan's native spacing).
    `SpectreImageFeatureExtractor` applies all of that for you when handed a raw scan in
    Hounsfield Units.

    Compatibility runs both ways:
    * A `config.json` written before the `preset` field existed carries an explicit
      `backbone_name` and `backbone_kwargs`; those take precedence, so it keeps loading its own
      architecture instead of silently inheriting this class's defaults.
    * Older code reading a newer `config.json` ignores the fields it does not know and reads the
      resolved architecture, which is still written out in full.
    """

    model_type = "spectre"

    def __init__(
        self,
        preset="spectre-large",
        # What the model expects to be fed.
        crop_size=DEFAULT_CROP_SIZE,
        hu_range=DEFAULT_HU_RANGE,
        voxel_spacing=None,
        orientation="RAS",
        # Architecture. Left as None these are resolved from `preset`; passing them explicitly
        # (as every pre-0.2 config.json does) takes precedence.
        backbone_name=None,
        backbone_kwargs=None,
        feature_combiner_name=None,
        feature_combiner_kwargs=None,
        include_feature_combiner=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.preset = preset
        self.crop_size = tuple(crop_size)
        self.hu_range = tuple(hu_range)
        self.voxel_spacing = tuple(voxel_spacing) if voxel_spacing is not None else None
        self.orientation = orientation
        self.include_feature_combiner = include_feature_combiner

        if backbone_name is None:
            resolved = get_preset(preset)
            backbone_name = resolved.backbone
            if backbone_kwargs is None:
                backbone_kwargs = resolved.backbone_kwargs
            feature_combiner_name = feature_combiner_name or resolved.feature_combiner
            if feature_combiner_kwargs is None:
                feature_combiner_kwargs = resolved.feature_combiner_kwargs

        self.backbone_name = backbone_name
        self.backbone_kwargs = backbone_kwargs or {}
        self.feature_combiner_name = feature_combiner_name if include_feature_combiner else None
        self.feature_combiner_kwargs = feature_combiner_kwargs or {}
