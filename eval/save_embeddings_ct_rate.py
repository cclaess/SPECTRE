import argparse
from pathlib import Path
from functools import partial

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
    Spacingd,
    ResizeWithPadOrCropd,
    GridPatchd,
)
from transformers import (
    Qwen2TokenizerFast, 
    Qwen3Model, 
    Qwen3Config,
)

import spectre.models as models
from spectre.data import CTRateDataset
from spectre.ssl.heads import SigLIPProjectionHead
from spectre.transforms import RandomReportTransformd
from spectre.utils import (
    extended_collate_siglip, 
    add_lora_adapters, 
    last_token_pool,
)


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Save embeddings from 3D NIfTI images using Spectre models"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, 
        help="Directory to CT-RATE dataset",
    )
    parser.add_argument(
        "--save_dir", type=str, default="embeddings", 
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--image_size", type=int, nargs=3, default=(512, 512, 384), 
        help="Size of the 3D image (H, W, D)",
    )
    parser.add_argument(
        "--patch_size", type=int, nargs=3, default=(128, 128, 64), 
        help="Size of the 3D patches (H, W, D)",
    )

    parser.add_argument(
        "--architecture", type=str, default="vit_large_patch16_128", 
        help="Model architecture for image backbone",
    )
    parser.add_argument(
        "--feature_comb_embed_dim", type=int, default=1080, 
        help="Embedding dimension for image feature combiner",
    )
    parser.add_argument(
        "--feature_comb_num_layers", type=int, default=4, 
        help="Number of layers in the image feature combiner",
    )
    parser.add_argument(
        "--feature_comb_num_heads", type=int, default=12, 
        help="Number of attention heads in the image feature combiner",
    )
    parser.add_argument(
        "--projection_dim", type=int, default=512, 
        help="Dimension of the projection layer for image features",
    )
    parser.add_argument(
        "--text_tokenizer", type=str, default="Qwen/Qwen3-Embedding-0.6B", 
        help="Tokenizer for text backbone",
    )
    parser.add_argument(
        "--use_lora", type=bool, default=True,
        help="Use LoRA adapters for the text backbone",
    )
    parser.add_argument(
        "--lora_r", type=int, default=16,
        help="Rank for LoRA adapters",
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=64.0,
        help="Alpha for LoRA adapters",
    )
    parser.add_argument(
        "--lora_target_keywords", type=str, nargs="+", 
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target keywords for LoRA adapters",
    )
    parser.add_argument(
        "--image_backbone_weights", type=str, default=None, 
        help="Path to the image backbone weights",
    )
    parser.add_argument(
        "--image_feature_comb_weights", type=str, default=None, 
        help="Path to the image feature combiner weights",
    )
    parser.add_argument(
        "--image_projection_weights", type=str, default=None, 
        help="Path to the image projection weights",
    )
    parser.add_argument(
        "--text_backbone_weights", type=str, default=None, 
        help="Path to the text backbone weights",
    )
    parser.add_argument(
        "--text_projection_weights", type=str, default=None, 
        help="Path to the text projection weights",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, 
        help="Batch size for the dataloader",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, 
        help="Number of workers for the dataloader",
    )
    return parser


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if the required weights are provided
    do_image_backbone = args.image_backbone_weights is not None
    do_image_feature_comb = do_image_backbone and args.image_feature_comb_weights is not None
    do_image_projection = do_image_feature_comb and args.image_projection_weights is not None
    do_text_backbone = args.text_backbone_weights is not None
    do_text_projection = do_text_backbone and args.text_projection_weights is not None

    if not (do_image_backbone or do_text_backbone):
        raise ValueError("At least one backbone (image or text) must be specified.")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define transformations for the dataset
    transform = [
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
        Spacingd(keys=("image",), pixdim=(0.5, 0.5, 1.0), mode=("bilinear",)),
        ResizeWithPadOrCropd(keys=("image",), spatial_size=args.image_size),
        GridPatchd(
            keys=("image",),
            patch_size=args.patch_size,
            overlap=0.0,
        ),
    ]
    if do_text_backbone:
        transform = Compose(
            transform + [
            RandomReportTransformd(
                keys=("findings", "impressions"),
                keep_original_prob=1.0,
                drop_prob=0.0,
                allow_missing_keys=False,
            )
        ])
    else:
        transform = Compose(transform)

    # Create dataset and dataloader
    dataset = CTRateDataset(
        data_dir=args.data_dir,
        include_reports=do_text_backbone,
        transform=transform,
        subset="valid",
        fraction=1.0,  # Use full validation set
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=partial(
            extended_collate_siglip, 
            tokenizer=Qwen2TokenizerFast.from_pretrained(
                args.text_tokenizer,
            ) if do_text_backbone else None,
            tokenizer_max_length=4096,
            return_filenames=True,
        ),
    )

    # Load the image backbone model
    if do_image_backbone:
        if (
            hasattr(models, args.architecture) 
            and args.architecture.startswith("vit")
        ):
            image_backbone = getattr(models, args.architecture)(
                pretrained_weights=args.image_backbone_weights,
                num_classes=0,
                global_pool='',  # Return all tokens
                pos_embed="rope",
                rope_kwargs={
                    "base": 1000.0,  # works for most 3D models
                },
                init_values=1.0,  # Will be set otherwise if present in weights
            )
            image_backbone_embed_dim = image_backbone.embed_dim
        else:
            raise NotImplementedError(f"Model {args.architecture} not implemented.")
        image_backbone.to(device).eval()
    
        # Load the image feature combiner if specified
        if do_image_feature_comb:
            image_feature_comb = models.FeatureVisionTransformer(
                patch_dim=image_backbone_embed_dim * 2,   # cls token + avg pooling (C. Jose et al. 2024)
                num_classes=0,
                global_pool='',
                embed_dim=args.feature_comb_embed_dim,
                depth=args.feature_comb_num_layers,
                num_heads=args.feature_comb_num_heads,
                pos_embed="rope",
                rope_kwargs={
                    "base": 100.0,
                },
                init_values=1.0,  # Will be set otherwise if present in weights
            )

            image_feature_comb.load_state_dict(
                torch.load(
                    args.image_feature_comb_weights, 
                    map_location="cpu", 
                    weights_only=False,
                ),
                strict=True,
            )
            image_feature_comb.to(device).eval()

            # Load the image projection model if specified
            if do_image_projection:
                image_projection = SigLIPProjectionHead(
                    input_dim=image_feature_comb.embed_dim * 2,  # cls token + avg pooling
                    output_dim=args.projection_dim,
                )
                image_projection.load_state_dict(
                    torch.load(
                        args.image_projection_weights, 
                        map_location="cpu", 
                        weights_only=False,
                    ),
                    strict=True,
                )
                image_projection.to(device).eval()
    
    # Load the text backbone model if specified
    if do_text_backbone:
        config = {
            "_attn_implementation_autoset": True,
            "architectures": [
                "Qwen3ForCausalLM"
            ],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151643,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 32768,
            "max_window_layers": 28,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "torch_dtype": "float32",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151669
        }
        text_backbone = Qwen3Model(Qwen3Config.from_dict(config))

        if args.use_lora and args.lora_r > 0:
            add_lora_adapters(
                text_backbone,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_keywords=args.lora_target_keywords,
            )

        text_backbone.load_state_dict(
            torch.load(
                args.text_backbone_weights, 
                map_location="cpu", 
                weights_only=False,
            ),
            strict=True,
        )
        text_backbone.to(device).eval()

        # Load the text projection model if specified
        if do_text_projection:
            text_projection = SigLIPProjectionHead(
                input_dim=text_backbone.config.hidden_size,
                output_dim=args.projection_dim,
            )
            text_projection.load_state_dict(
                torch.load(
                    args.text_projection_weights, 
                    map_location="cpu", 
                    weights_only=False,
                ),
                strict=True,
            )
            text_projection.to(device).eval()
    
    # Loop through the dataset and save embeddings
    for batch in tqdm(dataloader, desc="Processing batches"):
        # Move batch to device if is a tensor
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        filenames = [Path(f).name.split(".")[0] for f in batch["filename"]]
        save_paths = [save_dir / filename for filename in filenames]


        with torch.no_grad():
            if do_image_backbone:
                # Save the images as numpy arrays
                save_images(
                    batch["image"], 
                    [p / "image.npy" for p in save_paths]
                )

                B, N, C, H, W, D = batch["image"].shape
                images = batch["image"].view(B*N, C, H, W, D)  # Reshape to (B*N, C, H, W, D)

                image_embeddings = image_backbone(images)
                save_embeddings(
                    image_embeddings[:, 0].view(B, N, -1), 
                    [p / "image_backbone_cls.npy" for p in save_paths]
                )  # Save the CLS token embeddings of shape ()
                save_embeddings(
                    image_embeddings[:, 1:].view(B, N, image_embeddings.shape[1] - 1, -1),
                    [p / "image_backbone_patch.npy" for p in save_paths]
                )

                if do_image_feature_comb:
                    image_embeddings = image_embeddings.view(B, N, image_embeddings.shape[1], -1)  # (batch, crops, patches, embed_dim)
                    image_embeddings = torch.cat([
                        image_embeddings[:, :, 0, :],  # class token
                        image_embeddings[:, :, 1:, :].mean(dim=2)  # mean of patch tokens
                    ], dim=2)  # (batch, crops, embed_dim)
                    grid_size = tuple(
                        (args.image_size[i] // args.patch_size[i]) 
                        for i in range(3)
                    )  # (H, W, D)
                    image_embeddings = image_feature_comb(image_embeddings, grid_size=grid_size)
                    save_embeddings(
                        image_embeddings[:, 0, :], 
                        [p / "image_feature_comb_cls.npy" for p in save_paths]
                    )
                    save_embeddings(
                        image_embeddings[:, 1:, :], 
                        [p / "image_feature_comb_patch.npy" for p in save_paths]
                    )

                    if do_image_projection:
                        image_embeddings = torch.cat([
                            image_embeddings[:, 0, :],  # class token
                            image_embeddings[:, 1:, :].mean(dim=1)  # mean of patch tokens
                        ], dim=1)
                        image_embeddings = image_projection(image_embeddings)
                        save_embeddings(
                            image_embeddings, 
                            [p / "image_projection.npy" for p in save_paths]
                        )
                
            if do_text_backbone:
                text_embeddings = text_backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                text_embeddings = last_token_pool(
                    text_embeddings.last_hidden_state, batch["attention_mask"]
                )
                save_embeddings(
                    text_embeddings, 
                    [p / "text_backbone.npy" for p in save_paths]
                )
                if do_text_projection:
                    text_embeddings = text_projection(text_embeddings)
                    save_embeddings(
                        text_embeddings, 
                        [p / "text_projection.npy" for p in save_paths]
                    )


def save_embeddings(embeddings, save_paths):
    """
    Save embeddings to a file.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = torch.split(embeddings, 1, dim=0)
        embeddings = [emb.squeeze(0) for emb in embeddings if emb.numel() > 0]
    elif isinstance(embeddings, list):
        embeddings = [emb for emb in embeddings if isinstance(emb, torch.Tensor) and emb.numel() > 0]
    else:
        raise ValueError("Embeddings must be a tensor or a list of tensors.")
    
    assert len(embeddings) > 0, "No valid embeddings to save."
    assert len(embeddings) == len(save_paths), "Number of embeddings and save paths must match."

    for emb, save_path in zip(embeddings, save_paths):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".npy")
        
        np.save(save_path, emb.cpu().numpy())


def save_images(images, save_paths):
    """
    Save images to a file.
    """
    if isinstance(images, torch.Tensor):
        images = torch.split(images, 1, dim=0)
        images = [img.squeeze(0) for img in images if img.numel() > 0]
    elif isinstance(images, list):
        images = [img for img in images if isinstance(img, torch.Tensor) and img.numel() > 0]
    else:
        raise ValueError("Images must be a tensor or a list of tensors.")
    
    assert len(images) > 0, "No valid images to save."
    assert len(images) == len(save_paths), "Number of images and save paths must match."

    for img, save_path in zip(images, save_paths):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".npy")
        
        np.save(save_path, img.cpu().numpy())


if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
