import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Resized,
)

from transformers import (
    AutoProcessor,
    AutoModel,
)

from spectre.data import CTRateDataset
from spectre.utils import collate_add_filenames
from spectre.transforms import RandomReportTransformd


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="embeddings")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=10)
    return parser


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    transform = Compose([
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=("image",),
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        Orientationd(keys=("image",), axcodes="RAS"),
        Resized(keys=("image",), spatial_size=(448, 448, -1), mode="trilinear"),
        RandomReportTransformd(
            keys=("findings", "impressions"),
            keep_original_prob=1.0,
            drop_prob=0.0,
            allow_missing_keys=False,
        )
    ])

    dataset = CTRateDataset(
        data_dir=args.data_dir,
        include_reports=True,
        transform=transform,
        subset="valid",
        fraction=1.0,
    )

    processor = AutoProcessor.from_pretrained("google/medsiglip-448")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_add_filenames,
    )

    model = AutoModel.from_pretrained("google/medsiglip-448").to(device).eval()

    for batch in tqdm(dataloader):

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        filenames = [Path(f).stem for f in batch["filename"]]
        save_paths = [save_dir / f for f in filenames]

        with torch.no_grad():

            images = batch["image"]  # (B, C, H, W, D)
            B, C, H, W, D = images.shape

            images = images.permute(0, 4, 1, 2, 3)  # (B, D, C, H, W)
            images = images.reshape(B * D, C, H, W)

            images = images.repeat(1, 3, 1, 1)

            slice_features = model.get_image_features(pixel_values=images)  # (B*D, F)

            slice_features = slice_features.view(B, D, -1)  # (B, D, feat_dim)
            image_features = slice_features.max(dim=1).values  # max-pool over slices (B, feat_dim)

            save_embeddings(
                image_features,
                [p / "image_projection.npy" for p in save_paths]
            )
            texts = processor(
                text=batch["report"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            text_features = model.get_text_features(
                input_ids=texts["input_ids"],
                attention_mask=texts["attention_mask"]
            )  # (B, feat_dim)

            save_embeddings(
                text_features,
                [p / "text_projection.npy" for p in save_paths]
            )


def save_embeddings(embeddings, save_paths):
    embeddings = torch.split(embeddings, 1, dim=0)
    for emb, path in zip(embeddings, save_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
