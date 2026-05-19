import random
from typing import Iterable
from itertools import repeat

import torch
import numpy as np

MONAI_IMPORT_ERROR = None
try:
    import monai
except ImportError as e:
    monai = None  # type: ignore
    MONAI_IMPORT_ERROR = e

def fix_random_seeds(seed: int = 31):
    """ 
    Fix random seeds.
    """
    if MONAI_IMPORT_ERROR is not None:
        raise ImportError(
            "MONAI is required to use fix_random_seeds but not installed. "
            "Please install MONAI to use this function."
        ) from MONAI_IMPORT_ERROR

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    monai.utils.set_determinism(seed=seed)


def _ntuple(n: int):
    """
    Helper function to create n-tuple.
    """
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
