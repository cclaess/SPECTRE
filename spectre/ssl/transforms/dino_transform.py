from copy import deepcopy
from typing import Tuple, Mapping, Hashable, Any, List

import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    CenterSpatialCropd,
    SpatialPadd,
    EnsureTyped,
    RandSpatialCropSamplesd,
    SelectItemsd,
    RandSpatialCropSamples,
    RandFlip,
    OneOf,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGaussianNoise,
    RandAdjustContrast,
    Resize,
    MapTransform,
    Randomizable,
    LazyTransform,
)

from spectre.transforms import RandScaleIntensityRange


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
        assert dtype in ["float16", "float32"], \
            "dtype must be either 'float16' or 'float32'"

        device = "cuda" if (use_gds and torch.cuda.is_available()) else "cpu"
        base_crop_size = tuple(
            int(sz * (1 / local_views_scale[0])) for sz in local_views_size
        )

        super().__init__([
            LoadImaged(keys=("image",)),
            EnsureChannelFirstd(
                keys=("image",), 
                channel_dim="no_channel"
            ),
            ScaleIntensityRanged(
                keys=("image",),
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=("image",), axcodes="RAS"),
            Spacingd(
                keys=("image",), 
                pixdim=(0.5, 0.5, 1.0),  # comply with newest scanners
                mode=("bilinear",),
            ),
            CenterSpatialCropd(
                keys=("image",), 
                roi_size=(512, 512, 384),
            ),
            SpatialPadd(
                keys=("image",),
                spatial_size=base_crop_size,
            ),
            EnsureTyped(
                keys=("image",), 
                dtype=getattr(torch, dtype), 
                device=device,
            ),
            RandSpatialCropSamplesd(
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
            SelectItemsd(
                keys=("image_global_views", "image_local_views"),
            ),
        ])


class DINORandomCropTransformd(Randomizable, MapTransform, LazyTransform):
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
        MapTransform.__init__(self, keys)
        LazyTransform.__init__(self, lazy)
        self.global_views_size = global_views_size
        self.local_views_size = local_views_size
        self.local_views_scale = local_views_scale
        self.num_local_views = num_local_views

        self.cropper_global = RandSpatialCropSamples(
            roi_size=tuple(int(local_views_scale[1] * sz) for sz in base_crop_size),
            num_samples=2,
            max_roi_size=base_crop_size,
            random_center=True,
            random_size=True,
            lazy=lazy,
        )
        self.cropper_local = RandSpatialCropSamples(
            roi_size=tuple(int(self.local_views_scale[0] * sz) for sz in base_crop_size),
            num_samples=num_local_views,
            max_roi_size=tuple(int(self.local_views_scale[1] * sz) for sz in base_crop_size),
            random_center=True,
            random_size=True,
            lazy=lazy,
        )

        self.resize_global = Resize(
            spatial_size=global_views_size,
            mode="trilinear",
            dtype=getattr(torch, dtype),  # worst case 0.1-0.3% error for fp16
            anti_aliasing=True,  # downsample ratios up to 2
            lazy=lazy,
        )
        self.resize_local = Resize(
            spatial_size=local_views_size,
            mode="trilinear",
            dtype=getattr(torch, dtype),  # worst case 0.1-0.3% error for fp16
            anti_aliasing=True,  # downsample ratios up to 4
            lazy=lazy,
        )

        self.augmentor = Compose([
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandFlip(spatial_axis=2, prob=0.5),
            OneOf([
                RandGaussianSharpen(
                    sigma1_x=(1.5, 2.5), sigma1_y=(1.5, 2.5), sigma1_z=(0.75, 1.25),
                    sigma2_x=(0.5, 1.0), sigma2_y=(0.5, 1.0), sigma2_z=(0.25, 0.5),
                    prob=0.25,
                ),
                RandGaussianSmooth(
                    sigma_x=(1.5, 2.5), sigma_y=(1.5, 2.5), sigma_z=(0.75, 1.25),
                    prob=0.25,
                ),
            ]),
            RandAdjustContrast(gamma=(0.9, 1.1), prob=0.25),
            RandGaussianNoise(std=0.1, sample_std=True, prob=0.25),
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
