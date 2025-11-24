from typing import Optional, Tuple, Union
import warnings
import numpy as np
import torch

from monai.transforms import Transform, Randomizable
from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.utils import convert_to_tensor, convert_data_type


class RandScaleIntensityRange(Randomizable, Transform):
    """
    Randomizable variant of ScaleIntensityRange that samples the input window
    (a_min, a_max) per-call using MONAI's RNG (self.R).

    Args:
        a_min: float OR (low, high) tuple to sample a_min from.
        a_max: float OR (low, high) tuple to sample a_max from.
        b_min: target range min (scalar) or None (no shift).
        b_max: target range max (scalar) or None (no scaling to upper bound).
        clip: whether to clip to [b_min, b_max] after scaling.
        dtype: output dtype (defaults to np.float32).
        prob: probability to apply the transform (default 1.0).
    """

    backend = ["torch", "numpy"]

    def __init__(
        self,
        a_min: Union[float, Tuple[float, float]],
        a_max: Union[float, Tuple[float, float]],
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: DtypeLike = np.float32,
        prob: float = 1.0,
    ) -> None:
        Transform.__init__(self)
        Randomizable.__init__(self)
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = bool(clip)
        self.dtype = dtype
        self.prob = float(prob)

        # sampled values populated by randomize()
        self._sampled_a_min: Optional[float] = None
        self._sampled_a_max: Optional[float] = None
        self.do: bool = True

    def _sample_if_range(self, val: Union[float, Tuple[float, float], None]) -> Optional[float]:
        """Sample scalar if `val` is a (low, high) tuple, else return scalar or None."""
        if val is None:
            return None
        if isinstance(val, (tuple, list)) and len(val) == 2:
            low, high = float(val[0]), float(val[1])
            return float(self.R.uniform(low, high))
        return float(val)

    def randomize(self, data: Optional[object] = None) -> None:
        # whether to apply this transform on this call
        self.do = (self.R.rand() < self.prob)
        if not self.do:
            self._sampled_a_min = None
            self._sampled_a_max = None
            return

        # sample a_min & a_max using MONAI RNG
        self._sampled_a_min = self._sample_if_range(self.a_min)
        self._sampled_a_max = self._sample_if_range(self.a_max)

        # make sure we have valid window
        if (self._sampled_a_min is not None) and (self._sampled_a_max is not None):
            if self._sampled_a_max <= self._sampled_a_min:
                # ensure a tiny positive width (you can alternatively swap)
                self._sampled_a_max = self._sampled_a_min + 1e-6

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply scaling using *sampled* a_min/a_max (if sampling enabled).
        Returns torch tensor (like original did via convert_data_type).
        """
        # sample values for this call
        self.randomize(None)
        if not self.do:
            # no-op: still return as tensor with desired dtype
            x = convert_to_tensor(img)
            out = convert_data_type(x, dtype=self.dtype)[0]
            return out

        a_min = self._sampled_a_min
        a_max = self._sampled_a_max
        b_min = self.b_min
        b_max = self.b_max

        # Fallback to original scalar values if not ranges and not sampled earlier
        if a_min is None:
            # if user passed scalar a_min but randomize didn't set (shouldn't happen), coerce
            if isinstance(self.a_min, (int, float)):
                a_min = float(self.a_min)
        if a_max is None:
            if isinstance(self.a_max, (int, float)):
                a_max = float(self.a_max)

        # convert input to tensor (preserves numpy/torch input)
        img_t = convert_to_tensor(img).float()
        eps = 1e-8

        # degenerate input range
        if a_min is None or a_max is None or abs(a_max - a_min) < eps:
            warnings.warn("Degenerate input window (a_min == a_max or not provided)", UserWarning)
            if b_min is None:
                out = img_t - (a_min if a_min is not None else 0.0)
            else:
                out = img_t - (a_min if a_min is not None else 0.0) + b_min
            if self.clip and (b_min is not None or b_max is not None):
                lo = b_min if b_min is not None else -float("inf")
                hi = b_max if b_max is not None else float("inf")
                out = torch.clamp(out, min=lo, max=hi)
            return convert_data_type(out, dtype=self.dtype)[0]

        # scale into [0,1]
        out = (img_t - a_min) / (a_max - a_min)

        # scale/shift to [b_min, b_max] according to what was provided
        if (b_min is not None) and (b_max is not None):
            out = out * (b_max - b_min) + b_min
        elif (b_min is not None) and (b_max is None):
            out = out + b_min
        elif (b_min is None) and (b_max is not None):
            out = out * b_max

        if self.clip:
            lo = b_min if b_min is not None else -float("inf")
            hi = b_max if b_max is not None else float("inf")
            out = torch.clamp(out, min=lo, max=hi)

        return convert_data_type(out, dtype=self.dtype)[0]
