import os
import shutil
import argparse
import tempfile
from time import time
from functools import partial

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    EnsureTyped,
    RandSpatialCropd,
    DeleteItemsd,
)
from monai.utils import set_determinism
from transformers import Qwen2TokenizerFast

from spectre.utils import (
    get_dataloader,
    extended_collate_siglip,
)
from spectre.transforms import GenerateReportTransform


def get_args_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Test DataLoader Performance")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data",
        help="path to the directory containing the data",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="path to the directory for caching datasets",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="maximum number of steps to run",
    )
    parser.add_argument(
        "--keep_cache",
        action="store_true",
        help="whether to keep the cache directory after evaluation",
    )
    parser.add_argument(
        "--use_gds",
        action="store_true",
        help="whether to use GPU Direct Storage",
    )
    parser.add_argument(
        "--text_tokenizer",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="pretrained text tokenizer to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size for dataloaders",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers for dataloaders",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="whether to pin memory for dataloaders",
    )
    parser.add_argument(
        "--drop_last",
        action="store_true",
        help="whether to drop the last incomplete batch",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="whether to use persistent workers for dataloaders",
    )
    parser.add_argument(
        "--use_thread",
        action="store_true",
        help="whether to use threading for dataloaders",
    )
    return parser


def evaluate_dataloader(dataloader, max_steps):
    for epoch in range(2):
        start_time = time()
        num_samples = 0
        step_start_time = time()
        for idx, batch in enumerate(dataloader):
            if idx >= max_steps:
                break
            num_samples += batch["image"].size(0)
            print(f"Epoch {epoch:03} Step {idx:03}: {time() - step_start_time:.4f}s")
            step_start_time = time()
        end_time = time()
        print(f"Epoch {epoch:03}: {end_time - start_time:.4f}s, {num_samples / (end_time - start_time):.2f} samples/s")


def main(args):

    # Set seeds for reproducability
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=0)

    if args.cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="monai_cache_")
    else:
        cache_dir = args.cache_dir
    cache_dir_img_only = os.path.join(cache_dir, "image_only")
    cache_dir_img_text = os.path.join(cache_dir, "image_text")

    transforms = [
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=("image",),
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=("image",), axcodes="RAS"),
        Spacingd(keys=("image",), pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
        ResizeWithPadOrCropd(keys=("image",), spatial_size=(512, 512, 320)),
        EnsureTyped(keys=("image",), dtype=torch.float16, device="cuda" if args.use_gds else "cpu"),
        RandSpatialCropd(
            keys=("image",),
            roi_size=(384, 384, 256),
            random_center=True,
            random_size=False,
        ),
    ]
    transforms_img_only = Compose(transforms)
    transforms_img_text = Compose(transforms + [
        GenerateReportTransform(
            keys=("findings", "impressions", "icd10"),
            max_num_icd10=20,
            likelihood_original=0.5,
            drop_chance=0.3,
        ),
        DeleteItemsd(keys=("findings", "impressions", "icd10")),
    ])

    collate_fn = partial(
        extended_collate_siglip,
        tokenizer=Qwen2TokenizerFast.from_pretrained(
            args.text_tokenizer,
        )
    )
    dataloader_img_only = get_dataloader(
        "merlin",  # has `Impressions`, `Findings`, and `ICD-10`
        args.data_dir,
        include_reports=False,
        include_labels=False,
        cache_dataset=True,
        cache_dir=cache_dir_img_only,
        use_gds=args.use_gds,
        transform=transforms_img_only,
        fraction=0.01,  # test on a small subset
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        collate_fn=None,
        drop_last=args.drop_last,
        persistent_workers=args.persistent_workers,
        use_thread=args.use_thread,
    )
    dataloader_img_text = get_dataloader(
        "merlin",  # has `Impressions`, `Findings`, and `ICD-10`
        args.data_dir,
        include_reports=True,
        include_labels=False,
        cache_dataset=True,
        cache_dir=cache_dir_img_text,
        use_gds=args.use_gds,
        transform=transforms_img_text,
        fraction=0.05,  # test on a small subset
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=args.drop_last,
        persistent_workers=args.persistent_workers,
        use_thread=args.use_thread,
    )

    print("Image Only:")
    evaluate_dataloader(dataloader_img_only, args.max_steps)
    print("Image + Text:")
    evaluate_dataloader(dataloader_img_text, args.max_steps)

    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
