from pathlib import Path

from omegaconf import OmegaConf


def load_config(config_name: str) -> OmegaConf:
    """
    Load config file from path.
    """
    config_filename = config_name + ".yaml"
    config_path = Path(__file__).parent.resolve() / config_filename
    return OmegaConf.load(config_path)


default_config_dino = load_config("dino_default")
default_config_dinov2 = load_config("dinov2_default")
default_config_mae = load_config("mae_default")
default_config_siglip = load_config("siglip_default")


def load_and_merge_config(config_name: str, default_config: OmegaConf) -> OmegaConf:
    """
    Load and merge config file from path.
    """
    config = load_config(config_name)
    return OmegaConf.merge(default_config, config)