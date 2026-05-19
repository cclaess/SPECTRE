from __future__ import annotations

import warnings
from typing import Any
from pathlib import Path

_OMEGACONF_IMPORT_ERROR = None
try:
    from omegaconf import OmegaConf, DictConfig
except ImportError as e:
    OmegaConf, DictConfig = None, Any  # type: ignore
    _OMEGACONF_IMPORT_ERROR = e


def load_config(config_name: str) -> "DictConfig":
    """
    Load config file from path.
    """
    if _OMEGACONF_IMPORT_ERROR is not None:
        raise ImportError(
            "OmegaConf is required to load config files but not installed. "
            "Please install OmegaConf to use this feature."
        ) from _OMEGACONF_IMPORT_ERROR
    
    config_filename = config_name + ".yaml"
    config_path = Path(__file__).parent.resolve() / config_filename
    return OmegaConf.load(config_path)


if OmegaConf is not None:
    default_config_dino = load_config("dino_default")
    default_config_dinov2 = load_config("dinov2_default")
    default_config_mae = load_config("mae_default")
    default_config_siglip = load_config("siglip_default")
else:
    default_config_dino = None
    default_config_dinov2 = None
    default_config_mae = None
    default_config_siglip = None
    warnings.warn(
        "OmegaConf is not installed. Default configs will not be available and are set to `None`. "
        "Please install OmegaConf to use default configs."
    )


def load_and_merge_config(config_name: str, default_config: "DictConfig") -> "DictConfig":
    """
    Load and merge config file from path.
    """
    if _OMEGACONF_IMPORT_ERROR is not None:
        raise ImportError(
            "OmegaConf is required to load config files but not installed. "
            "Please install OmegaConf to use this feature."
        ) from _OMEGACONF_IMPORT_ERROR

    config = load_config(config_name)
    return OmegaConf.merge(default_config, config)
