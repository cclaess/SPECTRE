from copy import deepcopy
from typing import Tuple, Mapping, Hashable, Any, List

import torch

MONAI_IMPORT_ERROR = None
try:
    import monai.transforms as transforms
    from monai.config import KeysCollection
except ImportError as e:
    transforms = None  # type: ignore
    KeysCollection = Any  # type: ignore
    MONAI_IMPORT_ERROR = e

from spectre.transforms import RandScaleIntensityRange


if transforms is not None:
    Compose = transforms.Compose
    _BaseClass = (
        transforms.Randomizable,
        transforms.MapTransform,
        transforms.LazyTransform,
    )
else:
    Compose = object  # type: ignore
    _BaseClass = object  # type: ignore


class DINOTransform(Compose):
    def __init__(
        self,
        num_base_patches: int = 16,  # number of "samples" to draw from one CT scan for I/O efficiency
        global_views_size: Tuple[int, int, int] = (128, 128, 64),
        local_views_size: Tuple[int, int, int] = (48, 48, 24),
        local_views_scale: Tuple[float, float] = (0.1875, 0.5),
        num_local_views: int = 8,
        dtype: str = "float32",
        use_gds: bool = False,
    ):
        if MONAI_IMPORT_ERROR is not None:
            raise ImportError(
                "MONAI is required to use DINOTransform but not installed. "
                "Please install MONAI to use this transform."
            ) from MONAI_IMPORT_ERROR
        
        assert dtype in ["float16", "float32"], \
            "dtype must be either 'float16' or 'float32'"

        device = "cuda" if (use_gds and torch.cuda.is_available()) else "cpu"
        base_crop_size = tuple(
            int(sz * (1 / local_views_scale[0])) for sz in local_views_size
        )

        super().__init__([
            transforms.LoadImaged(keys=("image",)),
            transforms.EnsureChannelFirstd(
                keys=("image",), 
                channel_dim="no_channel"
            ),
            transforms.ScaleIntensityRanged(
                keys=("image",),
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            transforms.Orientationd(keys=("image",), axcodes="RAS"),
            transforms.Spacingd(
                keys=("image",), 
                pixdim=(0.5, 0.5, 1.0),  # comply with newest scanners
                mode=("bilinear",),
            ),
            transforms.CenterSpatialCropd(
                keys=("image",), 
                roi_size=(512, 512, 384),
            ),
            transforms.SpatialPadd(
                keys=("image",),
                spatial_size=base_crop_size,
            ),
            transforms.EnsureTyped(
                keys=("image",), 
                dtype=getattr(torch, dtype), 
                device=device,
            ),
            transforms.RandSpatialCropSamplesd(
                keys=("image",),
                num_samples=num_base_patches,
                roi_size=base_crop_size,
                random_size=False,
                random_center=True,
            ),
            DINORandomCropTransformd(
                keys=("image",),
                base_crop_size=base_crop_size,
                global_views_size=global_views_size,
                local_views_size=local_views_size,
                local_views_scale=local_views_scale,
                num_local_views=num_local_views,
                dtype=dtype,
            ),
            transforms.SelectItemsd(
                keys=("image_global_views", "image_local_views"),
            ),
        ])


class DINORandomCropTransformd(_BaseClass):
    def __init__(
        self,
        keys: KeysCollection,
        base_crop_size: Tuple[int, int, int] = (256, 256, 128),
        global_views_size: Tuple[int, int, int] = (128, 128, 64),
        local_views_size: Tuple[int, int, int] = (48, 48, 24),
        local_views_scale: Tuple[float, float] = (0.1875, 0.5),
        num_local_views: int = 8,
        dtype: str = "float32",
        lazy: bool = False,
    ) -> None:
        if MONAI_IMPORT_ERROR is not None:
            raise ImportError(
                "MONAI is required to use DINORandomCropTransformd but not installed. "
                "Please install MONAI to use this transform."
            ) from MONAI_IMPORT_ERROR
        
        transforms.MapTransform.__init__(self, keys)
        transforms.LazyTransform.__init__(self, lazy)
        self.global_views_size = global_views_size
        self.local_views_size = local_views_size
        self.local_views_scale = local_views_scale
        self.num_local_views = num_local_views

        self.cropper_global = transforms.RandSpatialCropSamples(
            roi_size=tuple(int(local_views_scale[1] * sz) for sz in base_crop_size),
            num_samples=2,
            max_roi_size=base_crop_size,
            random_center=True,
            random_size=True,
            lazy=lazy,
        )
        self.cropper_local = transforms.RandSpatialCropSamples(
            roi_size=tuple(int(self.local_views_scale[0] * sz) for sz in base_crop_size),
            num_samples=num_local_views,
            max_roi_size=tuple(int(self.local_views_scale[1] * sz) for sz in base_crop_size),
            random_center=True,
            random_size=True,
            lazy=lazy,
        )

        self.resize_global = transforms.Resize(
            spatial_size=global_views_size,
            mode="trilinear",
            dtype=getattr(torch, dtype),  # worst case 0.1-0.3% error for fp16
            anti_aliasing=True,  # downsample ratios up to 2
            lazy=lazy,
        )
        self.resize_local = transforms.Resize(
            spatial_size=local_views_size,
            mode="trilinear",
            dtype=getattr(torch, dtype),  # worst case 0.1-0.3% error for fp16
            anti_aliasing=True,  # downsample ratios up to 4
            lazy=lazy,
        )

        self.augmentor = transforms.Compose([
            transforms.RandFlip(spatial_axis=0, prob=0.5),
            transforms.RandFlip(spatial_axis=1, prob=0.5),
            transforms.RandFlip(spatial_axis=2, prob=0.5),
            transforms.OneOf([
                transforms.RandGaussianSharpen(
                    sigma1_x=(1.5, 2.5), sigma1_y=(1.5, 2.5), sigma1_z=(0.75, 1.25),
                    sigma2_x=(0.5, 1.0), sigma2_y=(0.5, 1.0), sigma2_z=(0.25, 0.5),
                    prob=0.25,
                ),
                transforms.RandGaussianSmooth(
                    sigma_x=(1.5, 2.5), sigma_y=(1.5, 2.5), sigma_z=(0.75, 1.25),
                    prob=0.25,
                ),
            ]),
            transforms.RandAdjustContrast(gamma=(0.9, 1.1), prob=0.25),
            transforms.RandGaussianNoise(std=0.1, sample_std=True, prob=0.25),
            RandScaleIntensityRange(
                a_min=(0.0, 0.4),  # [0.0 * 2000 - 1000, 0.4 * 2000 - 1000] = [-1000, -200]
                a_max=(0.6, 1.0),  # [0.6 * 2000 - 1000, 1.0 * 2000 - 1000] = [200, 1000]
                b_min=0.0,
                b_max=1.0,
                clip=True,
                prob=0.25,
            ),
        ], lazy=lazy)
    
    def randomize(self, data: Any = None) -> None:
        self.sub_seed = self.R.randint(0, 2**32 // 2 - 1)
        self.cropper_global.set_random_state(seed=self.sub_seed)
        self.cropper_local.set_random_state(seed=self.sub_seed)
        self.augmentor.set_random_state(seed=self.sub_seed)

    def __call__(
        self, 
        data: Mapping[Hashable, Any] | List[Mapping[Hashable, Any]], 
        lazy: bool | None = None,
    ) -> dict[Hashable, Any]:
        
        # support list of dicts as input
        if isinstance(data, list):
            return [self.__call__(d, lazy=lazy) for d in data]
        
        ret = dict()
        # deep copy all the unmodified data
        for key in set(data.keys()).difference(set(self.keys)):
            ret[key] = deepcopy(data[key])

        self.randomize()
        lazy_ = self.lazy if lazy is None else lazy

        for key in self.key_iterator(dict(data)):
            image = data[key]
            global_views = list(self.cropper_global(image, lazy=lazy_))
            local_views = list(self.cropper_local(image, lazy=lazy_))

            global_views = [self.resize_global(gv, lazy=lazy_) for gv in global_views]
            local_views = [self.resize_local(lv, lazy=lazy_) for lv in local_views]

            global_views = [self.augmentor(gv, lazy=lazy_) for gv in global_views]
            local_views = [self.augmentor(lv, lazy=lazy_) for lv in local_views]

            ret[f"{key}_global_views"] = global_views
            ret[f"{key}_local_views"] = local_views

        return ret
