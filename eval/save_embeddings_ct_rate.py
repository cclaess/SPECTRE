import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
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
)
from transformers import (
    XLMRobertaTokenizerFast, 
    XLMRobertaModel, 
    XLMRobertaConfig,
)

import spectre.models as models
from spectre.data import CTRateDataset
from spectre.utils.collate import extended_collate_siglip
from spectre.transforms import SWSpatialCropSamplesd, GenerateReportTransform


def get_args_parser():
    parser = argparse.ArgumentParser(description="Save embeddings from 3D NIfTI images using Spectre models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory to CT-RATE dataset")
    parser.add_argument("--save_dir", type=str, default="embeddings", help="Directory to save embeddings")
    parser.add_argument("--image_backbone_weights", type=str, required=True, help="Path to the image backbone weights")

    parser.add_argument("--architecture", type=str, default="vit_base_patch16_128", help="Model architecture for image backbone")
    parser.add_argument("--feature_comb_embed_dim", type=int, default=768, help="Embedding dimension for image feature combiner")
    parser.add_argument("--feature_comb_num_layers", type=int, default=4, help="Number of layers in the image feature combiner")
    parser.add_argument("--feature_comb_num_heads", type=int, default=12, help="Number of attention heads in the image feature combiner")
    parser.add_argument("--projection_dim", type=int, default=4096, help="Dimension of the projection layer for image features")
    parser.add_argument("--text_tokenizer", type=str, default="BAAI/bge-m3", help="Tokenizer for text backbone")

    parser.add_argument("--image_feature_comb_weights", type=str, default=None, help="Path to the image feature combiner weights")
    parser.add_argument("--image_projection_weights", type=str, default=None, help="Path to the image projection weights")
    parser.add_argument("--text_backbone_weights", type=str, default=None, help="Path to the text backbone weights")
    parser.add_argument("--text_projection_weights", type=str, default=None, help="Path to the text projection weights")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloader")
    return parser


def main(args):
    # Set some presets based on the input arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    do_image_feature_comb = args.image_feature_comb_weights is not None
    do_image_projection = do_image_feature_comb and args.image_projection_weights is not None
    do_text_backbone = args.text_backbone_weights is not None
    do_text_projection = do_text_backbone and args.text_projection_weights is not None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define transformations for the dataset
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-1000, 
            a_max=1000, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(384, 384, 256)),
        SWSpatialCropSamplesd(
            keys=["image"],
            patch_size=(128, 128, 64),
            overlap=0.0,
        ),
        GenerateReportTransform(
            keys=("findings", "impressions"),
            likelihood_original=1.0,
            drop_chance=0.0,
        )
    ])

    # Create dataset and dataloader
    dataset = CTRateDataset(
        data_dir=args.data_dir,
        include_reports=do_text_backbone,
        transform=transform,
        subset="valid",
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=partial(
            extended_collate_siglip, 
            tokenizer=XLMRobertaTokenizerFast.from_pretrained(
                args.text_tokenizer,
            ) if do_text_backbone else None,
            tokenizer_max_length=4096,
        ),
    )

    # Load the image backbone model
    if (
        hasattr(models, args.architecture) 
        and args.architecture.startswith("vit")
    ):
        image_backbone = getattr(models, args.architecture)(
            pretrained_weights=args.image_backbone_weights,
            num_classes=0,
        )
        image_backbone_embed_dim = image_backbone.embed_dim
    elif (
        hasattr(models, args.architecture)
        and args.architecture.startswith("resnet")
        or args.architecture.startswith("resnext")
    ):
        image_backbone = getattr(models, args.architecture)(
            pretrained_weights=args.image_backbone_weights,
            num_classes=0,
            norm_layer=partial(nn.BatchNorm3d, track_running_stats=False),
        )
        image_backbone_embed_dim = image_backbone.num_features
    else:
        raise NotImplementedError(f"Model {args.architecture} not implemented.")
    image_backbone.to(device).eval()
    
    # Load the image feature combiner if specified
    if do_image_feature_comb:
        image_feature_comb = models.FeatureVisionTransformer(
            patch_dim=image_backbone_embed_dim,
            embed_dim=args.feature_comb_embed_dim,
            num_patches=36,
            depth=args.feature_comb_num_layers,
            heads=args.feature_comb_num_heads,
        )

        image_feature_comb.load_state_dict(
            torch.load(
                args.image_feature_comb_weights, 
                map_location="cpu", 
                weights_only=False
            ),
            strict=True,
        )
        image_feature_comb.to(device).eval()

        # Load the image projection model if specified
        if do_image_projection:
            image_projection = nn.Linear(
                in_features=image_feature_comb.embed_dim,
                out_features=args.projection_dim,
            )
            image_projection.load_state_dict(
                torch.load(
                    args.image_projection_weights, 
                    map_location="cpu", 
                    weights_only=False
                ),
                strict=True,
            )
            image_projection.to(device).eval()
    
    # Load the text backbone model if specified
    if do_text_backbone:
        config = {
            "architectures": [
                "XLMRobertaModel"
            ],
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "classifier_dropout": None,
            "eos_token_id": 2,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 8194,
            "model_type": "xlm-roberta",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "output_past": True,
            "pad_token_id": 1,
            "position_embedding_type": "absolute",
            "torch_dtype": "float32",
            "transformers_version": "4.52.3",
            "type_vocab_size": 1,
            "use_cache": True,
            "vocab_size": 250002
        }
        
        text_backbone = XLMRobertaModel(XLMRobertaConfig.from_dict(config))

        text_backbone.load_state_dict(
            torch.load(
                args.text_backbone_weights, 
                map_location="cpu", 
                weights_only=False
            ),
            strict=False,
        )
        text_backbone.to(device).eval()

        # Load the text projection model if specified
        if do_text_projection:
            text_projection = nn.Linear(
                in_features=text_backbone.config.hidden_size,
                out_features=args.projection_dim,
            )
            text_projection.load_state_dict(
                torch.load(
                    args.text_projection_weights, 
                    map_location="cpu", 
                    weights_only=False
                ),
                strict=True,
            )
            text_projection.to(device).eval()
    
    # Loop through the dataset and save embeddings
    for batch in tqdm(dataloader, desc="Processing batches"):
        # Move batch to device if is a tensor
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        B, N, C, H, W, D = batch["image"].shape
        images = batch["image"].view(B*N, C, H, W, D)  # Reshape to (B*N, C, H, W, D)

        filenames = [Path(f).name.split(".")[0] for f in batch["filename"]]
        save_paths = [save_dir / filename for filename in filenames]

        with torch.no_grad():
            image_embeddings = image_backbone(images)
            save_embeddings(
                image_embeddings.view(B, N, -1), 
                [p / "image_backbone.npy" for p in save_paths]
            )

            if do_image_feature_comb:
                image_embeddings = image_feature_comb(image_embeddings.view(B, N, -1))
                save_embeddings(
                    image_embeddings, 
                    [p / "image_feature_comb.npy" for p in save_paths]
                )

                if do_image_projection:
                    image_embeddings = image_projection(image_embeddings)
                    save_embeddings(
                        image_embeddings, 
                        [p / "image_projection.npy" for p in save_paths]
                    )
            
            if do_text_backbone:
                text_embeddings = text_backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).pooler_output
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


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
