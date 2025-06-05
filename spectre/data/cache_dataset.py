import sys
import warnings
import pickle
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from monai.utils import look_up_option
from monai.data import PersistentDataset


SUPPORTED_PICKLE_MOD = {"pickle": pickle}


class CacheDataset(PersistentDataset):
    """
    Overwrite MONAI's PersistentDataset to support PyTorch 2.6.
    """
    def __init__(self, *args, pickle_protocol=pickle.HIGHEST_PROTOCOL, **kwargs):
        super().__init__(*args, pickle_protocol=pickle_protocol, **kwargs)
    
    def _cachecheck(self, item_transformed):
        """
        A function to cache the expensive input data transform operations
        so that huge data sets (larger than computer memory) can be processed
        on the fly as needed, and intermediate results written to disk for
        future use.

        Args:
            item_transformed: The current data element to be mutated into transformed representation

        Returns:
            The transformed data_element, either from cache, or explicitly computing it.

        Warning:
            The current implementation does not encode transform information as part of the
            hashing mechanism used for generating cache names when `hash_transform` is None.
            If the transforms applied are changed in any way, the objects in the cache dir will be invalid.

        """
        hashfile = None
        if self.cache_dir is not None:
            data_item_md5 = self.hash_func(item_transformed).decode("utf-8")
            data_item_md5 += self.transform_hash
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            try:
                # NOTE: We use `weights_only=False` to support PyTorch 2.6
                return torch.load(hashfile, weights_only=False)
            except PermissionError as e:
                if sys.platform != "win32":
                    raise e
            except RuntimeError as e:
                if "Invalid magic number; corrupt file" in str(e):
                    warnings.warn(f"Corrupt cache file detected: {hashfile}. Deleting and recomputing.")
                    hashfile.unlink()
                elif "PytorchStreamReader failed reading zip archive: failed finding central directory" in str(e):
                    warnings.warn(f"Corrupt cache file detected: {hashfile}. Deleting and recomputing.")
                    hashfile.unlink()
                else:
                    raise e

        _item_transformed = self._pre_transform(deepcopy(item_transformed))  # keep the original hashed
        
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_item_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(self.pickle_module, SUPPORTED_PICKLE_MOD),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed
