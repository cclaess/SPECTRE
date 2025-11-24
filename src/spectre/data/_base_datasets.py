import sys
import warnings
import pickle
import shutil
import tempfile
from typing import Any
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
import monai.data as data
from monai.utils import look_up_option, convert_to_tensor

try:
    import cupy as cp
    import kvikio.numpy as kvikio_numpy
except ImportError:
    cp = None
    kvikio_numpy = None


SUPPORTED_PICKLE_MOD = {"pickle": pickle}


class PersistentDataset(data.PersistentDataset):
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

        try:
            _item_transformed = self._pre_transform(deepcopy(item_transformed))  # keep the original hashed
        except RuntimeError as e:
            if "applying transform" in str(e):
                warnings.warn(f"Transform failed for item {item_transformed}. Skipping this item.")
                return self._cachecheck(self.data[0])  # Return the first item as a fallback


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


class GDSDataset(data.GDSDataset):
    """
    Overwrite MONAI's GDSDataset to support PyTorch 2.6 and combined GPU/CPU data (image/text pairs)
    without breaking the GDS fast path.
    """
    def __init__(self, *args, pickle_protocol=pickle.HIGHEST_PROTOCOL, **kwargs):
        super().__init__(*args, pickle_protocol=pickle_protocol, **kwargs)
    
    def _cachecheck(self, item_transformed):
        """
        In order to enable direct storage to the GPU when loading the hashfile, rewritten this function.
        Note that in this function, it will always return `torch.Tensor` when load data from cache.

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
        # compute a cache id
        if self.cache_dir is not None:
            data_item_md5 = self.hash_func(item_transformed).decode("utf-8")
            data_item_md5 += self.transform_hash
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            with cp.cuda.Device(self.device):
                if isinstance(item_transformed, dict):
                    item: dict[Any, Any] = {}
                    for k in item_transformed:
                        try:
                            meta_k = self._load_meta_cache(meta_hash_file_name=f"{hashfile.name}-{k}-meta")
                        except FileNotFoundError:
                            continue  # non-tensor key handled by sidecar
                        item[k] = kvikio_numpy.fromfile(f"{hashfile}-{k}", dtype=meta_k["dtype"], like=cp.empty(()))
                        item[k] = convert_to_tensor(item[k].reshape(meta_k["shape"]), device=f"cuda:{self.device}")
                        item[f"{k}_meta_dict"] = meta_k

                    sidecar_path = f"{hashfile}-aux"
                    aux: dict[str, Any] = {}
                    if Path(sidecar_path).is_file():
                        aux = torch.load(sidecar_path, weights_only=False)
                        item.update(aux)      
                    return item

                    # return item
                elif isinstance(item_transformed, (np.ndarray, torch.Tensor)):
                    _meta = self._load_meta_cache(meta_hash_file_name=f"{hashfile.name}-meta")
                    _data = kvikio_numpy.fromfile(f"{hashfile}", dtype=_meta["dtype"], like=cp.empty(()))
                    _data = convert_to_tensor(_data.reshape(_meta["shape"]), device=f"cuda:{self.device}")
                    filtered_keys = list(filter(lambda key: key not in ["dtype", "shape"], _meta.keys()))
                    if bool(filtered_keys):
                        return (_data, _meta)
                    return _data
                else:
                    item: list[dict[Any, Any]] = [{} for _ in range(len(item_transformed))]  # type:ignore
                    for i, _item in enumerate(item_transformed):
                        for k in _item:
                            meta_i_k = self._load_meta_cache(meta_hash_file_name=f"{hashfile.name}-{k}-meta-{i}")
                            item_k = kvikio_numpy.fromfile(
                                f"{hashfile}-{k}-{i}", dtype=meta_i_k["dtype"], like=cp.empty(())
                            )
                            item_k = convert_to_tensor(item[i].reshape(meta_i_k["shape"]), device=f"cuda:{self.device}")
                            item[i].update({k: item_k, f"{k}_meta_dict": meta_i_k})
                    return item

        # create new cache
        _item_transformed = self._pre_transform(deepcopy(item_transformed))  # keep the original hashed
        if hashfile is None:
            return _item_transformed
        if isinstance(_item_transformed, dict):
            # collect non-tensor fields into a sidecar so we don't lose them
            aux: dict[str, Any] = {}
            for k in _item_transformed:
                data_hashfile = f"{hashfile}-{k}"
                meta_hash_file_name = f"{hashfile.name}-{k}-meta"
                if isinstance(_item_transformed[k], (np.ndarray, torch.Tensor)):
                    self._create_new_cache(_item_transformed[k], data_hashfile, meta_hash_file_name)
                else:
                    aux[k] = _item_transformed[k]
            
            # write sidecar only if needed
            if aux:
                self._create_sidecar_cache(aux, f"{hashfile}-aux")
        elif isinstance(_item_transformed, (np.ndarray, torch.Tensor)):
            data_hashfile = f"{hashfile}"
            meta_hash_file_name = f"{hashfile.name}-meta"
            self._create_new_cache(_item_transformed, data_hashfile, meta_hash_file_name)
        else:
            for i, _item in enumerate(_item_transformed):
                for k in _item:
                    data_hashfile = f"{hashfile}-{k}-{i}"
                    meta_hash_file_name = f"{hashfile.name}-{k}-meta-{i}"
                    self._create_new_cache(_item, data_hashfile, meta_hash_file_name)
        open(hashfile, "a").close()  # store cacheid
        return _item_transformed
    
    def _create_sidecar_cache(self, aux_dict, sidecar_path):
        sidecar_hashfile = Path(sidecar_path)
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / sidecar_hashfile.name
                torch.save(
                    obj=aux_dict,
                    f=temp_hash_file,
                    pickle_module=look_up_option(self.pickle_module, SUPPORTED_PICKLE_MOD),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not sidecar_hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), sidecar_hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass

    def _load_meta_cache(self, meta_hash_file_name):
        if meta_hash_file_name in self._meta_cache:
            return self._meta_cache[meta_hash_file_name]
        else:
            return torch.load(self.cache_dir / meta_hash_file_name, weights_only=False)
