import argparse
from pathlib import Path
from functools import partial
import sys
import os
import pandas as pd

# --- 1. Set Backend for Headless Servers ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# -----------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

# --- IMPORTS FOR SEGMENTATION ---
from totalsegmentator.python_api import totalsegmentator

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    GridPatchd,
    MapTransform,
    SpatialCrop,
)

from spectre.transforms import LargestMultipleCenterCropd
from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
from spectre.utils import extended_collate_siglip

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
# 3. DATASET (Images + Coords only) (Fixed)
# -------------------------------------------------------------------------
class ImageOnlyNiftiDataset(Dataset):
    def __init__(self, data_dir: str, df: pd.DataFrame, transform=None):
        image_paths = list(Path(data_dir).glob("*.nii.gz"))
        
        data = []
        for image_path in image_paths:
            p_id = image_path.name.split("_")[0] 
            
            # Default Sentinel (Avoids NoneType error)
            center = [-1, -1, -1]
            
            # Match with DataFrame
            id_col = 'patient_id' 
            
            if id_col in df.columns:
                matched_row = df[df[id_col].astype(str) == p_id]
                if not matched_row.empty:
                    row = matched_row.iloc[0]
                    
                    # Extract Center
                    cx = row.get('LungCenter_X')
                    # Check if valid number and not the error code -1
                    if pd.notna(cx) and cx != -1:
                        center = [
                            int(row['LungCenter_X']),
                            int(row['LungCenter_Y']),
                            int(row['LungCenter_Z'])
                        ]
            
            data.append({
                "image": str(image_path), 
                "filename": str(image_path),
                "lung_center": center  # Now guaranteed to be a list, never None
            })
            
        super().__init__(data=data, transform=transform)
# -------------------------------------------------------------------------
# 4. MAIN
# -------------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser(description="Save image embeddings (No Text)")
    parser.add_argument("--data_dir", type=str, default=r"/home/20203686/own dataset")
    parser.add_argument("--save_dir", type=str, default=r"/home/20203686/crop embed2 fixed")
    parser.add_argument("--excel_path", type=str, default=r"/home/20203686/LBxSF_dataframe.xlsx")
    parser.add_argument("--patch_size", type=int, nargs=3, default=(128, 128, 64))
    parser.add_argument("--model_variant", type=str, default="spectre-large-pretrained", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--num_workers", type=int, default=0) 
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. PRE-PROCESS EXCEL (Fill in missing centers) ---
    df = ensure_centers_in_excel(args.data_dir, args.excel_path)

    # --- 2. PIPELINE ---
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
    dataset = ImageOnlyNiftiDataset(args.data_dir, df=df, transform=transform)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=partial(
            extended_collate_siglip, 
            tokenizer=None,  # No tokenizer = Image only collate
            tokenizer_max_length=4096,
            return_filenames=True,
        ),
    )

    # --- 3. LOAD IMAGE MODEL ---
    print(f"Loading Spectre Model: {args.model_variant}...")
    config = MODEL_CONFIGS[args.model_variant]
    image_model = SpectreImageFeatureExtractor.from_config(config)
    image_model.to(device).eval()

    # --- 4. INFERENCE ---
    print("\nStarting Inference...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        input_images = batch["image"].to(device)
        
        filenames = [Path(f).name.split(".")[0] for f in batch["filename"]]
        batch_save_paths = [save_dir / filename for filename in filenames]

        with torch.no_grad():
            image_features = image_model(input_images, grid_size=(4, 4, 6))
            save_embeddings(image_features, batch_save_paths)

def save_embeddings(image_embs, save_paths):
    if isinstance(image_embs, torch.Tensor):
        image_embs = [x.squeeze(0).cpu().numpy() for x in torch.split(image_embs, 1, dim=0)]
        
    for img_emb, path in zip(image_embs, save_paths):
        path = Path(path)
        patient_id = path.name.split("_")[0]
        # Save directly in parent directory with patient_id prefix
        output_file = path.parent / f"{patient_id}_embedding.npy"
        np.save(output_file, img_emb)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)