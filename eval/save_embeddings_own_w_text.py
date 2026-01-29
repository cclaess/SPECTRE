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
# 3. PRE-PROCESSING
# -------------------------------------------------------------------------

def get_lung_center(nifti_path):
    try:
        nifti_img = nib.load(nifti_path)
        # Fast mode
        seg = totalsegmentator(nifti_img, output=None, roi_subset=['lung_left', 'lung_right'], fast=True, quiet=True, ml=True)
        if not np.any(seg.get_fdata() > 0):
             # Fallback
             seg = totalsegmentator(nifti_img, output=None, roi_subset=['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right'], fast=False, quiet=True, ml=True)
        
        data = seg.get_fdata()
        if np.any(data > 0):
            indices = np.argwhere(data > 0)
            return ((indices.min(0) + indices.max(0)) // 2).astype(int).tolist()
        return None
    except Exception:
        return None

def ensure_centers_in_excel(data_dir, excel_path):
    print(f"Checking Excel: {excel_path}")
    df = pd.read_excel(excel_path)
    for c in ['LungCenter_X','LungCenter_Y','LungCenter_Z']: 
        if c not in df.columns: df[c] = np.nan
    
    df['patient_id'] = df['patient_id'].astype(str)
    paths = list(Path(data_dir).glob("*.nii.gz"))
    dirty = False

    for p in tqdm(paths, desc="Checking Centers"):
        pid = p.name.split("_")[0]
        mask = df['patient_id'] == pid
        if not mask.any(): continue
        idx = df.index[mask][0]

        if pd.isna(df.at[idx, 'LungCenter_X']):
            center = get_lung_center(str(p))
            if center:
                df.at[idx,'LungCenter_X'], df.at[idx,'LungCenter_Y'], df.at[idx,'LungCenter_Z'] = center
            else:
                df.at[idx,'LungCenter_X'] = -1
            dirty = True
            if dirty and idx%5==0: df.to_excel(excel_path, index=False)
    
    if dirty: df.to_excel(excel_path, index=False)
    return df

class CropLungsFromCoordsd(MapTransform):
    def __init__(self, keys, spatial_size=(512, 512, 384)):
        super().__init__(keys)
        self.spatial_size = np.array(spatial_size)

    def __call__(self, data):
        d = dict(data)
        c = d.get('lung_center', [-1,-1,-1])
        center = np.array(c) if (c is not None and not np.array_equal(c, [-1,-1,-1])) else np.array(d[self.keys[0]].shape[1:])//2
        
        roi_start = center - self.spatial_size // 2
        roi_end = roi_start + self.spatial_size
        cropper = SpatialCrop(roi_start=tuple(roi_start.astype(int)), roi_end=tuple(roi_end.astype(int)))
        
        for k in self.keys: d[k] = cropper(d[k])
        d.pop('lung_center', None)
        return d

# -------------------------------------------------------------------------
# 4. DATASET
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
# 5. MAIN PIPELINE
# -------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"/home/20203686/own dataset")
    parser.add_argument("--save_dir", type=str, default=r"/home/20203686/crop embed w text")
    parser.add_argument("--excel_path", type=str, default=r"/home/20203686/LBxSF_dataframe.xlsx")
    
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
    
    # Crop & Patch Params
    parser.add_argument("--crop_size", type=int, nargs=3, default=(512, 512, 384))
    parser.add_argument("--patch_size", type=int, nargs=3, default=(128, 128, 64))

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--precompute_centers", action="store_true", 
                        help="Run lung center computation before creating dataset")
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_centers_in_excel(args.data_dir, args.excel_path)
    print(f"✅ Lung centers ready for {len(df)} patients")

    
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.text_tokenizer)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.5, 0.5, 1.0), mode=("bilinear",)),
        CropLungsFromCoordsd(keys=["image"], spatial_size=(512,512,384)),      
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        LargestMultipleCenterCropd(keys=("image",), patch_size=args.patch_size),
        GridPatchd(keys=("image",), patch_size=args.patch_size, overlap=0.0),
    ])

    dataset = MultimodalNiftiDataset(args.data_dir, df, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=partial(
            extended_collate_siglip, 
            tokenizer=tokenizer, 
            tokenizer_max_length=4096, 
            return_filenames=True
        )
    )

    # -------------------------------------------------------------------------
    # 2. IMAGE MODEL
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("STEP 2: Loading Image Model...")
    print("=" * 70)
    print(f"Loading Spectre Model: {args.model_variant}...")
    config = MODEL_CONFIGS[args.model_variant]
    image_model = SpectreImageFeatureExtractor.from_config(config)
    image_model.to(device).eval()
    print(f"Image Model loaded")
    
    image_projection = SigLIPProjectionHead(
        input_dim=2160,
        output_dim=args.projection_dim,
    )
    image_projection.load_state_dict(
        torch.load(args.image_projection_weights, map_location="cpu", weights_only=False),
        strict=True,
    )
    image_projection.to(device).eval()
    print(f"✅ Image Projection loaded")

    # -------------------------------------------------------------------------
    # 3. TEXT MODEL
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("STEP 3: Loading Text Model...")
    print("=" * 70)
    
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
    print(f"✅ Text Backbone loaded")

    text_projection = SigLIPProjectionHead(
        input_dim=text_backbone.config.hidden_size,
        output_dim=args.projection_dim,
    )
    text_projection.load_state_dict(
        torch.load(args.text_projection_weights, map_location="cpu", weights_only=False),
        strict=True,
    )
    text_projection.to(device).eval()
    print(f"✅ Text Projection loaded")

    print("=" * 70)
    print("STEP 4: Starting Inference...")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # 4. INFERENCE LOOP (UPDATED TO SAVE RAW FEATURES)
    # -------------------------------------------------------------------------
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
            image_embeddings = image_embeddings / (image_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

            # Text Embeddings
            text_embeddings = text_backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            text_embeddings = last_token_pool(
                text_embeddings.last_hidden_state, batch["attention_mask"]
            )
            text_embeddings = text_projection(text_embeddings)
            text_embeddings = text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        # Save embeddings
        for i, fname in enumerate(filenames):
            patient_dir = save_dir / fname
            patient_dir.mkdir(exist_ok=True)

            np.save(patient_dir / "image_projection.npy", image_embeddings[i].cpu().numpy())
            np.save(patient_dir / "image_raw_concatenated.npy", concatenated_features[i].cpu().numpy())
            np.save(patient_dir / "text_projection.npy", text_embeddings[i].cpu().numpy())

    print("✅ All embeddings saved successfully!")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)