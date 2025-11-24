import argparse
from pathlib import Path
from functools import partial
from typing import List, Dict, Optional, Tuple

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
    GridPatchd,
)
from transformers import (
    Qwen2TokenizerFast, 
    Qwen3Model, 
    Qwen3Config,
)

import spectre.models as models
from spectre.data import MerlinDataset
from spectre.ssl.heads import SigLIPProjectionHead
from spectre.transforms import RandomReportTransformd, LargestMultipleCenterCropd
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
        help="Directory to MERLIN dataset",
    )
    parser.add_argument(
        "--save_dir", type=str, default="embeddings", 
        help="Directory to save embeddings",
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
        "--batch_size", type=int, default=1, 
        help="Batch size for the dataloader",
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, 
        help="Number of workers for the dataloader",
    )
    return parser


def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """
    Return the first index i where haystack[i:i+len(needle)] == needle, or None.
    Works with lists of ints.
    """
    if not needle:
        return None
    hn = len(haystack)
    nn = len(needle)
    if nn > hn:
        return None
    # naive search (ok for typical token lengths)
    for i in range(hn - nn + 1):
        if haystack[i:i+nn] == needle:
            return i
    return None


def split_batch_by_headers(
    tokenizer,
    batch_input_ids: torch.LongTensor,
    batch_attention_mask: torch.LongTensor,
    headers: List[str] = ["Findings:", "Impressions:", "ICD10:"],
    output_pad: bool = False,
    pad_token_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Optional[torch.LongTensor]]], Optional[Dict[str, torch.LongTensor]]]:
    """
    Split a batch of tokenized sequences into sections based on header tokens.

    Args:
        tokenizer: HuggingFace tokenizer (used only to encode the header strings).
        batch_input_ids: (B, S) LongTensor from dataloader.
        batch_attention_mask: (B, S) LongTensor.
        headers: list of header strings in the order you expect them.
        output_pad: if True, also return padded tensors for each section across the batch.
        pad_token_id: token id to use for padding if output_pad True. If None, uses tokenizer.pad_token_id.

    Returns:
        per_example_sections: list of length B; each item is a dict with keys:
            - "findings_ids", "findings_mask", "impressions_ids", "impressions_mask", "icd10_ids", "icd10_mask"
            Each value is either a LongTensor (L,) or None if that section wasn't found.
        padded_outputs (optional): dict mapping section name -> padded tensor (B, Lmax) if output_pad True,
            and section_mask -> (B, Lmax). Otherwise None.
    """
    device = batch_input_ids.device
    B = batch_input_ids.shape[0]

    # Encode headers without special tokens
    header_token_ids = {h: tokenizer(h, add_special_tokens=False)["input_ids"] for h in headers}
    # Map header short names for output keys
    header_keys = [h.rstrip(":").lower() for h in headers]  # e.g. "Findings:" -> "findings"

    if output_pad:
        padded = {}
        for k in header_keys:
            padded[f"{k}_ids"] = []
            padded[f"{k}_mask"] = []
    else:
        padded = None

    # choose pad token id
    if output_pad and pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    per_example = []
    for b in range(B):
        input_ids = batch_input_ids[b].tolist()
        attn = batch_attention_mask[b].tolist()

        # compute actual seq length from attention_mask (ignore padded tokens)
        seq_len = int(sum(attn))
        if seq_len == 0:
            # empty example (shouldn't happen), create empty sections
            example = {f"{k}_ids": None for k in header_keys}
            for k in header_keys:
                example[f"{k}_mask"] = None
            per_example.append(example)
            if output_pad:
                for k in header_keys:
                    padded[f"{k}_ids"].append(torch.tensor([], dtype=torch.long))
                    padded[f"{k}_mask"].append(torch.tensor([], dtype=torch.long))
            continue

        eof_id = tokenizer.eos_token_id or input_ids[seq_len - 1]  # last active token is assumed to be EOF/eos
        
        # active tokens excludes the EOF token (we will append it per section)
        active_ids = input_ids[: seq_len - 1]  # exclude final eos token
        active_mask = attn[: seq_len - 1]

        # find header positions
        found_positions = {}
        for header in headers:
            pos = _find_subsequence(active_ids, header_token_ids[header])
            if pos is not None:
                found_positions[header] = pos

        sorted_headers = sorted(found_positions.items(), key=lambda x: x[1])  # [(header, pos), ...]

        # create slices per header
        example = {}
        for header in headers:
            key = header.rstrip(":").lower()
            if header not in found_positions:
                example[f"{key}_ids"] = None
                example[f"{key}_mask"] = None
                if output_pad:
                    padded[f"{key}_ids"].append(torch.tensor([], dtype=torch.long))
                    padded[f"{key}_mask"].append(torch.tensor([], dtype=torch.long))
                continue

            start = found_positions[header]
            # next header start if any, otherwise to end of active_ids
            next_pos_list = [pos for _, pos in sorted_headers if pos > start]
            end = next_pos_list[0] if next_pos_list else len(active_ids)

            # slice (this includes header tokens + content)
            ids_slice = active_ids[start:end]
            mask_slice = active_mask[start:end]

            # append EOF token and a 1 in the attention mask to indicate end-of-sequence
            ids_with_eof = ids_slice + [eof_id]
            mask_with_eof = mask_slice + [1]

            ids_t = torch.tensor(ids_with_eof, dtype=torch.long, device=device)
            mask_t = torch.tensor(mask_with_eof, dtype=torch.long, device=device)

            example[f"{key}_ids"] = ids_t
            example[f"{key}_mask"] = mask_t

            if output_pad:
                padded[f"{key}_ids"].append(ids_t.cpu())
                padded[f"{key}_mask"].append(mask_t.cpu())

        per_example.append(example)

    # If requested, convert padded lists into padded tensors (B, Lmax) per section
    padded_outputs = None
    if output_pad:
        padded_outputs = {}
        for k in header_keys:
            ids_list: List[torch.Tensor] = padded[f"{k}_ids"]
            mask_list: List[torch.Tensor] = padded[f"{k}_mask"]
            # compute max length
            max_len = max([t.numel() for t in ids_list], default=0)
            # Prepare padded tensors
            ids_padded = torch.full((B, max_len), pad_token_id, dtype=torch.long)
            mask_padded = torch.zeros((B, max_len), dtype=torch.long)
            for i, (ids_t, mask_t) in enumerate(zip(ids_list, mask_list)):
                L = ids_t.numel()
                if L > 0:
                    ids_padded[i, :L] = ids_t
                    mask_padded[i, :L] = mask_t
            padded_outputs[f"{k}_ids"] = ids_padded.to(device)
            padded_outputs[f"{k}_mask"] = mask_padded.to(device)

    return per_example, padded_outputs


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
        LargestMultipleCenterCropd(
            keys=("image",),
            patch_size=args.patch_size,
        ),
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
                max_num_icd10=-1,  # Use all ICD10 codes
                keep_original_prob=1.0,
                drop_prob=0.0,
                allow_missing_keys=False,
            )
        ])
    else:
        transform = Compose(transform)

    # Create dataset and dataloader
    dataset = MerlinDataset(
        data_dir=args.data_dir,
        include_reports=do_text_backbone,
        transform=transform,
        subset="test",
        fraction=1.0,  # Use full test set
    )
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.text_tokenizer) if do_text_backbone else None
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=partial(
            extended_collate_siglip, 
            tokenizer=tokenizer,
            tokenizer_max_length=None,
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
                checkpoint_path_or_url=args.image_backbone_weights,
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

        # Check if all embeddings already exist for this batch
        all_exist = True
        for p in save_paths:
            if not p.joinpath("image.npy").exists():
                all_exist = False
            if do_image_backbone:
                if not p.joinpath("image_backbone_cls.npy").exists() or not p.joinpath("image_backbone_patch.npy").exists():
                    all_exist = False
                if do_image_feature_comb:
                    if not p.joinpath("image_feature_comb_cls.npy").exists() or not p.joinpath("image_feature_comb_patch.npy").exists():
                        all_exist = False
                    if do_image_projection:
                        if not p.joinpath("image_projection.npy").exists():
                            all_exist = False
            if do_text_backbone:
                if not p.joinpath("text_backbone.npy").exists():
                    all_exist = False
                if do_text_projection:
                    if not p.joinpath("text_projection.npy").exists():
                        all_exist = False
        if all_exist:
            continue  # Skip this batch

        with torch.no_grad():
            if do_image_backbone:

                loc = batch["image"].data.meta["location"][0]
                Hp, Wp, Dp = tuple(int(np.unique(loc[i, :]).size) for i in range(3))
                B, N, C, H, W, D = batch["image"].shape

                assert N == (Hp * Wp * Dp), \
                    f"Number of patches {N} does not match computed grid size {Hp}x{Wp}x{Dp}"

                # Save the images as numpy arrays
                images_for_saving = batch["image"].view(B, Hp, Wp, Dp, C, H, W, D)
                images_for_saving = images_for_saving.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
                images_for_saving = images_for_saving.view(B, C, Hp * H, Wp * W, Dp * D)
                save_images(
                    images_for_saving, 
                    [p / "image.npy" for p in save_paths]
                )
                images = batch["image"].view(B*N, C, H, W, D)  # Reshape to (B*N, C, H, W, D)

                image_embeddings = image_backbone(images)
                save_embeddings(
                    image_embeddings[:, 0].view(B, Hp, Wp, Dp, -1), 
                    [p / "image_backbone_cls.npy" for p in save_paths]
                )  # Save the CLS token embeddings of shape (B, Hp, Wp, Dp, embed_dim)
                save_embeddings(
                    image_embeddings[:, 1:].view(B, Hp, Wp, Dp, image_embeddings.shape[1] - 1, -1),
                    [p / "image_backbone_patch.npy" for p in save_paths]
                )

                if do_image_feature_comb:
                    image_embeddings = image_embeddings.view(B, N, image_embeddings.shape[1], -1)  # (batch, crops, patches, embed_dim)
                    image_embeddings = torch.cat([
                        image_embeddings[:, :, 0, :],  # class token
                        image_embeddings[:, :, 1:, :].mean(dim=2)  # mean of patch tokens
                    ], dim=2)  # (batch, crops, embed_dim)
                    image_embeddings = image_feature_comb(image_embeddings, grid_size=(Hp, Wp, Dp))
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

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                _, padded = split_batch_by_headers(
                    tokenizer=tokenizer,
                    batch_input_ids=input_ids,
                    batch_attention_mask=attention_mask,
                    headers=["Findings:", "Impressions:", "ICD10:"],
                    output_pad=True,
                )

                text_embeddings = last_token_pool(text_backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).last_hidden_state, batch["attention_mask"])

                text_embeddings_findings = last_token_pool(text_backbone(
                    input_ids=padded["findings_ids"],
                    attention_mask=padded["findings_mask"]
                ).last_hidden_state, padded["findings_mask"])

                text_embeddings_impressions = last_token_pool(text_backbone(
                    input_ids=padded["impressions_ids"],
                    attention_mask=padded["impressions_mask"]
                ).last_hidden_state, padded["impressions_mask"])

                save_embeddings(
                    text_embeddings, 
                    [p / "text_backbone.npy" for p in save_paths]
                )
                save_embeddings(
                    text_embeddings_findings,
                    [p / "text_backbone_findings.npy" for p in save_paths]
                )
                save_embeddings(
                    text_embeddings_impressions,
                    [p / "text_backbone_impressions.npy" for p in save_paths]
                )
                if do_text_projection:
                    text_embeddings = text_projection(text_embeddings)
                    text_embeddings_findings = text_projection(text_embeddings_findings)
                    text_embeddings_impressions = text_projection(text_embeddings_impressions)
                    save_embeddings(
                        text_embeddings, 
                        [p / "text_projection.npy" for p in save_paths]
                    )
                    save_embeddings(
                        text_embeddings_findings,
                        [p / "text_projection_findings.npy" for p in save_paths]
                    )
                    save_embeddings(
                        text_embeddings_impressions,
                        [p / "text_projection_impressions.npy" for p in save_paths]
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
