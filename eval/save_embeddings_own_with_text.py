import argparse
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from pathlib import Path
from functools import partial
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# Update this path if needed to point to your project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from totalsegmentator.python_api import totalsegmentator
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Orientationd, Spacingd, GridPatchd, MapTransform, SpatialCrop,
)

# Spectre
from spectre.transforms import LargestMultipleCenterCropd, RandomReportTransformd
from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
from spectre.ssl.heads import SigLIPProjectionHead
from spectre.utils import extended_collate_siglip, last_token_pool, add_lora_adapters

# Transformers
from transformers import Qwen2TokenizerFast, Qwen3Model, Qwen3Config

# -------------------------------------------------------------------------
# 1. PRE-PROCESSING: Calculate Centers and Update Excel
# -------------------------------------------------------------------------
def get_lung_center(nifti_path):
    """
    Runs TotalSegmentator on a single file and returns the (x, y, z) center.
    Returns None if segmentation fails.
    """
    try:
        # Load NIfTI
        nifti_img = nib.load(nifti_path)
        
        # Run Fast Segmentation
        try:
            seg_nifti = totalsegmentator(
                nifti_img, 
                output=None, 
                roi_subset=['lung_left', 'lung_right'], 
                fast=True, 
                quiet=True,
                ml=True
            )
            seg_data = seg_nifti.get_fdata()
            
            # Retry Standard if empty
            if not np.any(seg_data > 0):
                raise ValueError("Fast segmentation empty")

        except Exception:
            # Fallback to Standard
            seg_nifti = totalsegmentator(
                nifti_img, 
                output=None, 
                roi_subset=['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right'], 
                fast=False,
                quiet=True,
                ml=True
            )
            seg_data = seg_nifti.get_fdata()

        if np.any(seg_data > 0):
            foreground_indices = np.argwhere(seg_data > 0)
            min_coords = foreground_indices.min(axis=0)
            max_coords = foreground_indices.max(axis=0)
            center = (min_coords + max_coords) // 2
            return center.astype(int).tolist() # [x, y, z]
        else:
            print(f"  Warning: No lungs found in {Path(nifti_path).name}")
            return None

    except Exception as e:
        print(f"  Error processing {nifti_path}: {e}")
        return None

def ensure_centers_in_excel(data_dir, excel_path, id_col='patient_id'):
    """
    Iterates through images. If center is missing in Excel, calculates it and saves immediately.
    """
    print(f"Checking Excel for lung centers: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # Ensure columns exist
    for col in ['LungCenter_X', 'LungCenter_Y', 'LungCenter_Z']:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure ID column is string for matching
    df[id_col] = df[id_col].astype(str)
    
    image_paths = list(Path(data_dir).glob("*.nii.gz"))
    dirty = False
    
    for img_path in tqdm(image_paths, desc="Verifying/Computing Centers"):
        p_id = img_path.name.split("_")[0]
        
        # Find row
        mask = df[id_col] == p_id
        if not mask.any():
            continue
            
        idx = df.index[mask][0]
        
        # Check if we already have coords
        if pd.isna(df.at[idx, 'LungCenter_X']):
            center = get_lung_center(img_path)
            
            if center is not None:
                df.at[idx, 'LungCenter_X'] = center[0]
                df.at[idx, 'LungCenter_Y'] = center[1]
                df.at[idx, 'LungCenter_Z'] = center[2]
                dirty = True
                
                # Save every 5 updates to be safe
                if dirty and (idx % 5 == 0):
                    df.to_excel(excel_path, index=False)
            else:
                # Mark as -1 to avoid re-trying failed files endlessly
                df.at[idx, 'LungCenter_X'] = -1
                dirty = True

    if dirty:
        print("Saving updated Excel with new centers...")
        df.to_excel(excel_path, index=False)
    else:
        print("Excel is up to date.")
    
    return df

# -------------------------------------------------------------------------
# 2. TRANSFORM: Crop using Pre-Calculated Coords (Fixed)
# -------------------------------------------------------------------------
class CropLungsFromCoordsd(MapTransform):
    """
    Crops image based on coordinates provided in the data dictionary.
    """
    def __init__(self, keys, spatial_size=(512, 512, 384)):
        super().__init__(keys)
        self.spatial_size = np.array(spatial_size)

    def __call__(self, data):
        d = dict(data)
        
        # 1. Retrieve the center (sentinel is [-1, -1, -1])
        raw_center = d.get('lung_center', [-1, -1, -1])
        
        # 2. Determine if valid
        # We treat [-1, -1, -1] as invalid/missing
        is_valid_center = (
            raw_center is not None 
            and isinstance(raw_center, (list, tuple, np.ndarray))
            and len(raw_center) == 3
            and not np.array_equal(raw_center, [-1, -1, -1])
        )

        for key in self.keys:
            image_tensor = d[key] # (C, H, W, D)
            
            if is_valid_center:
                center = np.array(raw_center)
            else:
                # Fallback to image center
                center = np.array(image_tensor.shape[1:]) // 2
            
            # --- Perform Crop Calculation ---
            roi_start = center - self.spatial_size // 2
            roi_end = roi_start + self.spatial_size
            
            # Handle padding
            spatial_shape = np.array(image_tensor.shape[1:])
            pad_lower = np.maximum(-roi_start, 0)
            pad_upper = np.maximum(roi_end - spatial_shape, 0)
            
            current_data = image_tensor 
            
            if np.any(pad_lower > 0) or np.any(pad_upper > 0):
                pads = [(0, 0)] + list(zip(pad_lower, pad_upper))
                current_data_np = current_data.cpu().numpy()
                current_data_np = np.pad(current_data_np, pads, mode='constant', constant_values=-1000)
                current_data = torch.from_numpy(current_data_np)
                roi_start = roi_start + pad_lower
            
            roi_end = roi_start + self.spatial_size
            roi_start_tuple = tuple(roi_start.astype(int).tolist())
            roi_end_tuple = tuple(roi_end.astype(int).tolist())

            cropper = SpatialCrop(roi_start=roi_start_tuple, roi_end=roi_end_tuple)
            d[key] = cropper(current_data)

        # 3. CRITICAL FIX: Remove 'lung_center' from dict
        # This prevents the DataLoader from trying to collate/stack it, 
        # avoiding the "unsupported type [None]" error entirely.
        d.pop('lung_center', None)
        
        return d

# -------------------------------------------------------------------------
# 3. DATASET
# -------------------------------------------------------------------------

class MultimodalNiftiDataset(Dataset):
    def __init__(self, data_dir, df, transform=None):
        data = []
        
        # Theoretical clinical cut-off values from Quorsane et al., 2025
        CUTOFF_CEA = 5.0    # ng/mL
        CUTOFF_CYFRA = 3.3  # ng/mL

        for i, p in enumerate(Path(data_dir).glob("*.nii.gz")):
            # FIX: Ensure we split by underscore to find the ID
            pid = p.name.split("_")[0]
            center = [-1, -1, -1]
            
            # Default text if no match found
            text = "Clinical data not available."
            
            row = df[df['patient_id'].astype(str) == pid]
            
            if not row.empty:
                r = row.iloc[0]
                if pd.notna(r.get('LungCenter_X')) and r.get('LungCenter_X') != -1:
                    center = [int(r['LungCenter_X']), int(r['LungCenter_Y']), int(r['LungCenter_Z'])]
                
                # Demographics
                gender = "male" if r.get('Gender') == 1 else "female" if r.get('Gender') == 0 else "patient"
                age = r.get('Age', 'unknown')

                # CYFRA 21-1
                cyfra_val = r.get('CYFRA21-1')
                if pd.notna(cyfra_val):
                    cyfra_status = "elevated" if float(cyfra_val) > CUTOFF_CYFRA else "considered normal"
                    cyfra_segment = (f"The CYFRA 21-1 levels are {cyfra_val} ng/mL, "
                                     f"which are {cyfra_status} as the cutoff value is {CUTOFF_CYFRA} ng/mL.")
                else:
                    cyfra_segment = "The CYFRA 21-1 levels are unknown."
                
                # CEA
                cea_val = r.get('CEA')
                if pd.notna(cea_val):
                    cea_status = "elevated" if float(cea_val) > CUTOFF_CEA else "considered normal"
                    cea_segment = (f"The CEA levels are {cea_val} ng/mL, "
                                   f"which are {cea_status} as the cutoff value is {CUTOFF_CEA} ng/mL.")
                else:
                    cea_segment = "The CEA levels are unknown."

                ctdna_raw = r.get('ctDNA')
                if ctdna_raw == 1:
                    ctdna_segment = "Furthermore, the liquid biopsy is positive for ctDNA, indicating the presence of driver mutations in EGFR, KRAS, or BRAF."
                elif ctdna_raw == 0:
                    ctdna_segment = "Furthermore, the liquid biopsy is negative for ctDNA, indicating an absence of driver mutations in EGFR, KRAS, or BRAF."
                else:
                    ctdna_segment = "Furthermore, the ctDNA status regarding driver mutations is unknown."

                text = (
                    f"This patient is a {age} year old {gender}. "
                    f"{cyfra_segment} "
                    f"{cea_segment} "
                    f"{ctdna_segment}"
                )
            
            # FIX: Provide ALL keys to ensure the pipeline finds the text
            data.append({
                "image": str(p), 
                "filename": str(p.name), 
                "lung_center": center, 
                "text": text,       # For generic use
                "findings": text,   # For RandomReportTransformd
                "impressions": "",  # For RandomReportTransformd
                "report": text      # Just in case
            })

            if i < 3:
                print(f"\n[DEBUG] Narrative Report for {pid}:")
                print(f"   \"{text}\"")

        super().__init__(data=data, transform=transform)

# -------------------------------------------------------------------------
# 4. MAIN
# -------------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser(description="Save image embeddings (No Text)")
    parser.add_argument("--data_dir", type=str, default=r"/home/20203686/own dataset")
    parser.add_argument("--save_dir", type=str, default=r"/home/20203686/crop embed w text 2")
    parser.add_argument("--excel_path", type=str, default=r"/home/20203686/LBxSF_dataframe.xlsx")
    parser.add_argument("--patch_size", type=int, nargs=3, default=(128, 128, 64))
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--num_workers", type=int, default=0) 

    # Image Model Params
    parser.add_argument("--model_variant", type=str, default="spectre-large-pretrained", 
                        help="Spectre model variant from MODEL_CONFIGS")
    parser.add_argument("--image_projection_weights", type=str, default="/home/20203686/SigLIP_projection_head_image.pt")
    parser.add_argument("--projection_dim", type=int, default=512)

    # Text Model Params
    parser.add_argument("--text_tokenizer", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--text_backbone_weights", type=str, default="/home/20203686/spectre_qwen3_embedding_0.6B_lora.pt")
    parser.add_argument("--text_projection_weights", type=str, default="/home/20203686/SigLIP_projection_head_text.pt")
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    parser.add_argument("--lora_target_keywords", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_centers_in_excel(args.data_dir, args.excel_path)
    print(f" Lung centers ready for {len(df)} patients")

    # Load the tokenizer

    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.5, 0.5, 1.0), mode=("bilinear",)),
        
        # Uses coordinates from Excel
        CropLungsFromCoordsd(keys=["image"], spatial_size=(512,512,384)),
        
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        LargestMultipleCenterCropd(keys=("image",), patch_size=args.patch_size),
        GridPatchd(keys=("image",), patch_size=args.patch_size, overlap=0.0),
    ])

    print("Initializing Dataset...")
    dataset = MultimodalNiftiDataset(args.data_dir, df=df, transform=transform)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=partial(
            extended_collate_siglip, 
            tokenizer=Qwen2TokenizerFast.from_pretrained(
                args.text_tokenizer,
            ),
            tokenizer_max_length=4096,
            return_filenames=True,
        ),
    )

    # Image model
    print(f"Loading Spectre Model: {args.model_variant}...")
    config = MODEL_CONFIGS[args.model_variant]
    image_model = SpectreImageFeatureExtractor.from_config(config)
    image_model.to(device).eval()
    print("Image model loaded")

    image_projection = SigLIPProjectionHead(
        input_dim=2160,
        output_dim=args.projection_dim,
    )
    image_projection.load_state_dict(
        torch.load(args.image_projection_weights, map_location="cpu", weights_only=False),
        strict=True,
    )
    image_projection.to(device).eval()
    print(f"Image Projection loaded")

    # Text model
    config = {
        "_attn_implementation_autoset": True,
        "architectures": ["Qwen3ForCausalLM"],
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
        torch.load(args.text_backbone_weights, map_location="cpu", weights_only=False),
        strict=True,
    )
    text_backbone.to(device).eval()
    print(f"Text Backbone loaded")

    text_projection = SigLIPProjectionHead(
        input_dim=text_backbone.config.hidden_size,
        output_dim=args.projection_dim,
    )
    text_projection.load_state_dict(
        torch.load(args.text_projection_weights, map_location="cpu", weights_only=False),
        strict=True,
    )
    text_projection.to(device).eval()
    print(f"Text Projection loaded")


    # --- 4. INFERENCE ---
    print("\nStarting Inference...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        filenames = [Path(f).name.split(".")[0] for f in batch["filename"]]

        with torch.no_grad():
            image_features = image_model(batch["image"], grid_size=(4, 4, 6) )

            if len(image_features.shape) == 3:
                cls_token = image_features[:, 0, :]      # (B, 1080)
                patch_tokens = image_features[:, 1:, :]  # (B, N, 1080)
                mean_patches = patch_tokens.mean(dim=1)  # (B, 1080)
                concatenated_features = torch.cat([cls_token, mean_patches], dim=1) # (B, 2160)
            else:
                raise ValueError(f"Unexpected image_features shape: {image_features.shape}")

            image_embeddings = image_projection(concatenated_features)
            #image_embeddings = image_embeddings / (image_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

            # Text Embeddings
            text_embeddings = text_backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            text_embeddings = last_token_pool(
                text_embeddings.last_hidden_state, batch["attention_mask"]
            )
            text_embeddings = text_projection(text_embeddings)
            #text_embeddings = text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        # Save embeddings
        for i, fname in enumerate(filenames):
            patient_dir = save_dir / fname
            patient_dir.mkdir(exist_ok=True)

            np.save(patient_dir / "image_projection.npy", image_embeddings[i].cpu().numpy())
            np.save(patient_dir / "image_raw_concatenated.npy", concatenated_features[i].cpu().numpy())
            np.save(patient_dir / "text_projection.npy", text_embeddings[i].cpu().numpy())


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)