from copy import deepcopy
from typing import Sequence, List, Mapping, Hashable

import torch
import torch.nn.functional as F
from monai.transforms import Transform, MapTransform
from monai.utils import ensure_tuple_rep


class SWSpatialCropSamples(Transform):
    """
    Crop image into patches using a sliding window approach and return a list of cropped patches.

    This transform splits an input tensor into patches based on the specified patch size and overlap.
    The sliding window is deterministic, and patches are generated sequentially without randomization.

    Args:
        patch_size: The size of the patches to be generated. Can be an int or a sequence of ints for each dimension.
        overlap: The amount of overlap between patches. Can be a float in [0.0, 1.0) (fraction of patch size)
            or an int (number of pixels). Default: 0.0.
        pad_mode: Padding mode for out-of-bound areas. Options are `"constant"`, `"reflect"`, `"replicate"`, `"circular"`.
            If None, no padding is applied, and out-of-bound patches are dropped. Default: `"constant"`.
        pad_value: The constant value used for padding when `pad_mode="constant"`. Default: 0.
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int,
        overlap: Sequence[float] | float | Sequence[int] | int = 0.0,
        pad_mode: str | None = "constant",
        pad_value: float = 0,
    ) -> None:
        super().__init__()
        self.patch_size = ensure_tuple_rep(patch_size, 3)  # Default to 3D input
        self.overlap = ensure_tuple_rep(overlap, 3)
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def _calculate_pad(self, input_shape: Sequence[int]) -> Sequence[int]:
        """
        Calculate padding for each dimension based on input shape and patch size.
        """
        pad = [0] * (2 * len(input_shape))
        for i, (dim_size, patch, ov) in enumerate(zip(input_shape, self.patch_size, self.overlap)):
            if isinstance(ov, float):
                ov = int(patch * ov)
            stride = patch - ov
            total_coverage = ((dim_size + stride - 1) // stride) * stride
            pad_start = 0
            pad_end = max(0, total_coverage - dim_size)
            pad[i * 2] = pad_start
            pad[i * 2 + 1] = pad_end
        return pad

    def _generate_positions(self, input_shape: Sequence[int]) -> List[List[int]]:
        """
        Generate the starting positions for each patch based on input shape, patch size, and overlap.
        """
        positions = []
        for dim_size, patch, ov in zip(input_shape, self.patch_size, self.overlap):
            if isinstance(ov, float):
                ov = int(patch * ov)
            stride = patch - ov
            pos = list(range(0, dim_size - patch + 1, stride))
            if pos[-1] + patch < dim_size:
                pos.append(dim_size - patch)
            positions.append(pos)
        return positions

    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply the sliding window cropping to the input tensor.

        Args:
            img: A torch tensor of shape (C, H, W) or (C, H, W, D).

        Returns:
            List[torch.Tensor]: A list of cropped patches.
        """
        # Validate input dimensions
        spatial_shape = img.shape[1:]
        if len(spatial_shape) != len(self.patch_size):
            raise ValueError(
                f"Input spatial dimensions {len(spatial_shape)} do not match patch size {len(self.patch_size)}."
            )

        # Apply padding if required
        pad = self._calculate_pad(spatial_shape)
        if any(pad):
            img = F.pad(img, pad[::-1], mode=self.pad_mode, value=self.pad_value)

        # Generate sliding window positions
        positions = self._generate_positions(img.shape[1:])
        all_patches = []

        # Extract patches
        for z in positions[2] if len(positions) > 2 else [0]:  # 3D or 2D
            for y in positions[1]:
                for x in positions[0]:
                    slices = [slice(None), slice(x, x + self.patch_size[0]), slice(y, y + self.patch_size[1])]
                    if len(self.patch_size) == 3:
                        slices.append(slice(z, z + self.patch_size[2]))
                    all_patches.append(img[tuple(slices)])

        return all_patches


class SWSpatialCropSamplesd(MapTransform):
    """
    Dictionary-based version of SlidingWindowSpatialCropSamples.

    This transform applies a sliding window cropping strategy to the specified fields in a dictionary.
    The sliding window is deterministic and does not involve randomization. It generates a list of dictionaries
    where each dictionary contains a cropped patch for the specified keys, along with any additional metadata.

    Args:
        keys: Keys of the corresponding items to be transformed.
        patch_size: The size of the patches to be generated. Can be an int or a sequence of ints for each dimension.
        overlap: The amount of overlap between patches. Can be a float in [0.0, 1.0) (fraction of patch size)
            or an int (number of pixels). Default: 0.0.
        pad_mode: Padding mode for out-of-bound areas. Options are `"constant"`, `"reflect"`, `"replicate"`, `"circular"`.
            If None, no padding is applied, and out-of-bound patches are dropped. Default: `"constant"`.
        pad_value: The constant value used for padding when `pad_mode="constant"`. Default: 0.
        allow_missing_keys: Don't raise an exception if a key is missing.
    """

    def __init__(
        self,
        keys: Sequence[Hashable],
        patch_size: Sequence[int] | int,
        overlap: Sequence[float] | float | Sequence[int] | int = 0.0,
        pad_mode: str | None = "constant",
        pad_value: float = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.patch_size = ensure_tuple_rep(patch_size, 3)
        self.overlap = ensure_tuple_rep(overlap, 3)
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def _calculate_pad(self, input_shape: Sequence[int]) -> Sequence[int]:
        """
        Calculate padding for each dimension based on input shape and patch size.
        """
        pad = [0] * (2 * len(input_shape))
        for i, (dim_size, patch, ov) in enumerate(zip(input_shape, self.patch_size, self.overlap)):
            if isinstance(ov, float):
                ov = int(patch * ov)
            stride = patch - ov
            total_coverage = ((dim_size + stride - 1) // stride) * stride
            pad_start = 0
            pad_end = max(0, total_coverage - dim_size)
            pad[i * 2] = pad_start
            pad[i * 2 + 1] = pad_end
        return pad

    def _generate_positions(self, input_shape: Sequence[int]) -> Sequence[Sequence[int]]:
        """
        Generate the starting positions for each patch based on input shape, patch size, and overlap.
        """
        positions = []
        for dim_size, patch, ov in zip(input_shape, self.patch_size, self.overlap):
            if isinstance(ov, float):
                ov = int(patch * ov)
            stride = patch - ov
            pos = list(range(0, dim_size - patch + 1, stride))
            if pos[-1] + patch < dim_size:
                pos.append(dim_size - patch)
            positions.append(pos)
        return positions

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> list[Mapping[Hashable, torch.Tensor]]:
        """
        Apply the sliding window cropping to the specified keys in the dictionary.

        Args:
            data: A dictionary where each key contains a torch.Tensor.

        Returns:
            List[Mapping[Hashable, torch.Tensor]]: A list of dictionaries for all the cropped patches.
        """
        ret = []

        for key in self.key_iterator(data):
            img = data[key]
            spatial_shape = img.shape[1:]

            if len(spatial_shape) != len(self.patch_size):
                raise ValueError(
                    f"Input spatial dimensions {len(spatial_shape)} do not match patch size {len(self.patch_size)}."
                )

            # Apply padding if required
            pad = self._calculate_pad(spatial_shape)
            if any(pad):
                img = F.pad(img, pad[::-1], mode=self.pad_mode, value=self.pad_value)

            # Generate sliding window positions
            positions = self._generate_positions(img.shape[1:])

            # Extract patches
            all_patches = []
            for z in positions[2] if len(positions) > 2 else [0]:  # 3D or 2D
                for y in positions[1]:
                    for x in positions[0]:
                        slices = [slice(None), slice(x, x + self.patch_size[0]), slice(y, y + self.patch_size[1])]
                        if len(self.patch_size) == 3:
                            slices.append(slice(z, z + self.patch_size[2]))
                        all_patches.append(img[tuple(slices)])

            # Create output dictionaries
            for i, patch in enumerate(all_patches):
                patch_dict = {key: patch}
                for other_key in set(data.keys()).difference(set(self.keys)):
                    patch_dict[other_key] = deepcopy(data[other_key])
                ret.append(patch_dict)

        return ret
