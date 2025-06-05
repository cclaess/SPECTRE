import pickle

import torch
from monai.data import GDSDataset as GDSDatasetMONAI


class GDSDataset(GDSDatasetMONAI):
    """
    Overwrite MONAI's PersistentDataset to support PyTorch 2.6.
    """
    def __init__(self, *args, pickle_protocol=pickle.HIGHEST_PROTOCOL, **kwargs):
        super().__init__(*args, pickle_protocol=pickle_protocol, **kwargs)
    
    def _load_meta_cache(self, meta_hash_file_name):
        if meta_hash_file_name in self._meta_cache:
            return self._meta_cache[meta_hash_file_name]
        else:
            return torch.load(self.cache_dir / meta_hash_file_name, weights_only=False)
