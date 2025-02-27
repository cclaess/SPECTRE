import os
import math

from omegaconf import OmegaConf

from spectre.utils import distributed, utils


def apply_scaling_rules_to_cfg(cfg):
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args, default_config):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def random_seed(args):
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    utils.fix_random_seeds(seed + rank)


def setup(args, default_config):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args, default_config)
    os.makedirs(args.output_dir, exist_ok=True)
    random_seed(args)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
