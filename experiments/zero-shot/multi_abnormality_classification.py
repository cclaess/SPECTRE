import argparse
from functools import partial

import torch
from torch import nn
from accelerate import Accelerator
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
)
from transformers import XLMRobertaModel, XLMRobertaConfig, XLMRobertaTokenizerFast

import spectre.models as models
from spectre.transforms import SWSpatialCropSamplesd
from spectre.utils.config import setup
from spectre.utils.dataloader import get_dataloader
from spectre.utils.checkpointing import clean_components_from_checkpoint


def get_args_parser() -> argparse.ArgumentParser:
    """
    Load arguments from config file. If argument is specified in command line, 
    it will override the value in config file.
    """
    parser = argparse.ArgumentParser(description="Zero-Shot Multi Abnormality Classification")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/classification_default.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="command line arguments to override config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="output directory to save checkpoints and logs",
    )
    return parser


def main(cfg, accelerator: Accelerator):
    """
    Main function to run the zero-shot multi abnormality classification experiment.
    """
    # Print config
    accelerator.print(cfg)

    # Get transforms
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
        Spacingd(keys=("image",), pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
        ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, 256)),
        SWSpatialCropSamplesd(
            keys=("image",),
            patch_size=(128, 128, 64),
            overlap=0.0,
        ),

    ])

    # Get dataloader
    data_loader = get_dataloader(
        cfg.train.datasets,
        cfg.train.data_dir,
        include_reports=False,
        include_labels=True,
        cache_dataset=cfg.train.cache_dataset,
        cache_dir=cfg.train.cache_dir,
        use_gds=cfg.train.use_gds,
        transform=transform,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        shuffle=True,
        drop_last=cfg.train.drop_last,
        persistent_workers=cfg.train.persistent_workers,
    )

    # Initialize image backbone and feature combiner
    if (
        hasattr(models, cfg.model.architecture) 
        and cfg.model.architecture.startswith("vit")
    ):
        image_backbone = getattr(models, cfg.model.architecture)(
            pretrained_weights=cfg.model.pretrained_weights,
            num_classes=0,
        )
        image_backbone_embed_dim = image_backbone.embed_dim
    elif (
        hasattr(models, cfg.model.architecture)
        and cfg.model.architecture.startswith("resnet")
        or cfg.model.architecture.startswith("resnext")
    ):
        image_backbone = getattr(models, cfg.model.architecture)(
            pretrained_weights=cfg.model.pretrained_weights,
            num_classes=0,
            norm_layer=partial(nn.BatchNorm3d, track_running_stats=False),
        )
        image_embed_dim = image_backbone.num_features
    else:
        raise NotImplementedError(f"Model {cfg.model.architecture} not implemented.")

    image_feature_comb = models.FeatureVisionTransformer(
        patch_dim=image_backbone_embed_dim,
        embed_dim=cfg.model.feature_comb_embed_dim,
        num_patches=36,
        depth=cfg.model.feature_comb_num_layers,
        heads=cfg.model.feature_comb_num_heads,
    )

    # Initialize text backbone
    # TODO: add support for other text backbones
    # AutoModel is not yet compatible with newest Pytorch Docker image
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
    
    text_tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        cfg.model.text_tokenizer,
    )
    text_backbone = XLMRobertaModel(XLMRobertaConfig.from_dict(config))
    text_embed_dim = text_backbone.config.hidden_size

    # Initialize projection layers
    image_projection = nn.Linear(image_embed_dim, cfg.model.projection_dim)
    text_projection = nn.Linear(text_embed_dim, cfg.model.projection_dim)

    # Load pretrained weights
    if cfg.model.pretrained_weights:
        accelerator.print(f"Loading pretrained weights from {cfg.model.pretrained_weights}")
        state_dict = torch.load(cfg.model.pretrained_weights, map_location="cpu")
        state_dict = clean_components_from_checkpoint(state_dict)
        msg = image_backbone.load_state_dict(state_dict["image_backbone"], strict=False)
        accelerator.print(f"Image backbone loaded with message: {msg}")
        msg = text_backbone.load_state_dict(state_dict["text_backbone"], strict=False)
        accelerator.print(f"Text backbone loaded with message: {msg}")
        msg = image_feature_comb.load_state_dict(state_dict["image_feature_comb"], strict=False)
        accelerator.print(f"Image feature combiner loaded with message: {msg}")
        msg = image_projection.load_state_dict(state_dict["image_projection"], strict=False)
        accelerator.print(f"Image projection loaded with message: {msg}")
        msg = text_projection.load_state_dict(state_dict["text_projection"], strict=False)
        accelerator.print(f"Text projection loaded with message: {msg}")
    else:
        raise ValueError("Pretrained weights not specified. Please provide a valid path.")
    
    # Prepare models and dataloader for distributed evaluation
    image_backbone, image_feature_comb, text_backbone, image_projection, \
        text_projection, data_loader = accelerator.prepare(
        image_backbone, image_feature_comb, text_backbone, 
        image_projection, text_projection, data_loader
    )

    # Start evaluation
    image_backbone.eval()
    image_feature_comb.eval()
    text_backbone.eval()
    image_projection.eval()
    text_projection.eval()
    for batch in data_loader:
        with torch.no_grad():

            image_projections = image_projection(
                image_feature_comb(image_backbone(batch["image"]))
            )
            positive_tokens = text_tokenizer.batch_encode_plus(
                ["Lung nodule is present"] * batch["image"].shape[0],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=1024,
            )
            positive_tokens = {
                "input_ids": torch.tensor(positive_tokens["input_ids"], device=accelerator.device),
                "attention_mask": torch.tensor(positive_tokens["attention_mask"], device=accelerator.device),
            }
            positive_text_projections = text_projection(
                text_backbone(**positive_tokens).last_hidden_state[:, 0, :]
            )
            negative_tokens = text_tokenizer.batch_encode_plus(
                ["Lung nodule is not present"] * batch["image"].shape[0],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=1024,
            )
            negative_tokens = {
                "input_ids": torch.tensor(negative_tokens["input_ids"], device=accelerator.device),
                "attention_mask": torch.tensor(negative_tokens["attention_mask"], device=accelerator.device),
            }
            negative_text_projections = text_projection(
                text_backbone(**negative_tokens).last_hidden_state[:, 0, :]
            )

            # Compute cosine similarity
            positive_similarity = torch.cosine_similarity(
                image_projections, positive_text_projections, dim=-1
            )
            negative_similarity = torch.cosine_similarity(
                image_projections, negative_text_projections, dim=-1
            )

            # Pick labels based on similarity
            predictions = torch.where(
                positive_similarity > negative_similarity, 
                torch.tensor(1, device=accelerator.device), 
                torch.tensor(0, device=accelerator.device)
            )

            # Compute metrics
            correct_predictions = (predictions == batch["labels"]).sum().item()
            total_predictions = predictions.shape[0]
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            accelerator.print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    cfg, accelerator = setup(args, default_config_zero_shot_classification)
    main(cfg, accelerator)
