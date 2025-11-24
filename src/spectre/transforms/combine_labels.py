from typing import Sequence, Optional, Hashable, Mapping

import torch
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.config.type_definitions import NdarrayOrTensor


class CombineLabelsd(MapTransform):
    """
    Combine multiple label images into a single combined mask.

    Args:
        keys (list): List of keys for the associated label images in the dictionary.
        mask_key (str): Key to store the combined mask in the dictionary.
        labels (list or None): List of label values corresponding to each key. Default is None,
                                which assigns labels as 1, 2, 3, ... in the order of keys.
        allow_missing_keys (bool): If True, does not raise an exception if a key is missing.
    """
    def __init__(
        self, 
        keys: KeysCollection, 
        mask_key: str, 
        labels: Optional[Sequence[int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        
        if labels is not None and len(keys) != len(labels):
            raise ValueError("The number of keys must match the number of labels provided.")
        
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.mask_key = mask_key
        self.labels = labels

    def __call__(
        self, 
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Mapping[Hashable, NdarrayOrTensor]:
        
        d = dict(data)
        combined_mask = None

        for idx, key in enumerate(self.key_iterator(d)):
            if self.labels is not None:
                current_label = self.labels[idx]
            else:
                current_label = idx + 1

            if combined_mask is None and isinstance(d[key], torch.Tensor):
                combined_mask = torch.zeros_like(d[key], dtype=d[key].dtype)
            elif combined_mask is None and isinstance(d[key], np.ndarray):
                combined_mask = np.zeros_like(d[key], dtype=d[key].dtype)
            elif combined_mask is None:
                raise TypeError(
                    f"Unsupported type for key '{key}': {type(d[key])}. "
                    "Expected torch.Tensor or np.ndarray."
                )

            combined_mask[d[key] > 0] = current_label

            # Remove the original key from the dictionary
            d.pop(key)

        d[self.mask_key] = combined_mask

        return d
