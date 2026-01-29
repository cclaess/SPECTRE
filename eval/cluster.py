import argparse
from pathlib import Path
from functools import partial
import sys
import os

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
    BorderPad 
)

from spectre.transforms import LargestMultipleCenterCropd
from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
from spectre.utils import extended_collate_siglip

class CropLungsAlignedd(MapTransform):
    """
    Segments lungs using TotalSegmentator.
    - Tries 'fast' mode first.
    - If 'fast' mode fails (KeyError), retries with 'standard' mode (slower but robust).
    - NEVER falls back to full-body segmentation.
    """
    def __init__(self, keys, spatial_size=(512, 512, 384), fast=True, save_dir=None):
        super().__init__(keys)
        self.spatial_size = np.array(spatial_size)
        self.default_fast = fast
        self.save_dir = Path(save_dir) if save_dir else None

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image_tensor = d[key] # Expecting (C, H, W, D)
            filename = d.get('filename', 'unknown_file')
            
            # --- 1. Prepare for TotalSegmentator ---
            data_np = image_tensor.cpu().numpy().squeeze(0) 
            
            if hasattr(image_tensor, "affine"):
                affine = image_tensor.affine.numpy()
            else:
                affine = np.eye(4)
                
            nifti_img = nib.Nifti1Image(data_np, affine)
            
            # Initialize empty
            seg_data = np.zeros_like(data_np)
            center = None
            found_target = False

            # --- 2. Run Segmentation (Smart Retry) ---
            try:
                # ATTEMPT 1: User Preference (usually Fast)
                # print(f"Attempting lung segmentation (Fast={self.default_fast})...")
                seg_nifti = totalsegmentator(
                    nifti_img, 
                    output=None, 
                    roi_subset=['lung_left', 'lung_right'], 
                    fast=self.default_fast, 
                    quiet=True,
                    ml=True
                )
                seg_data = seg_nifti.get_fdata()
                
                if np.any(seg_data > 0):
                    found_target = True

            except KeyError:
                # This catches the 'lung_left' missing key error in Fast model
                print(f"Warning: 'lung_left' key missing in Fast model for {filename}. Switching to Standard model...")
                
                try:
                    # ATTEMPT 2: Standard Model (Robust, guaranteed to have lung keys)
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
                        found_target = True
                        print(f"Lungs found with Standard model.")
                except Exception as e:
                    print(f"Standard model also failed: {e}")

            except Exception as e:
                print(f"General segmentation error for {filename}: {e}")

            # --- 3. Check Success and Calculate Center ---
            if found_target:
                foreground_indices = np.argwhere(seg_data > 0)
                min_coords = foreground_indices.min(axis=0)
                max_coords = foreground_indices.max(axis=0)
                center = (min_coords + max_coords) // 2
            else:
                # Strict Failure if you prefer, or center fallback
                print(f"CRITICAL: No lungs found for {filename}. Using image center.")
                center = np.array(image_tensor.shape[1:]) // 2
                # Note: seg_data is still all zeros, so no red overlay will appear.

            # --- 4. Calculate Initial Crop Box ---
            roi_start = center - self.spatial_size // 2
            roi_end = roi_start + self.spatial_size
            
            # --- 5. VISUALIZATION ---
            if self.save_dir:
                try:
                    slice_idx = int(center[1])
                    
                    # Transpose for plotting (Z, X)
                    img_slice = data_np[:, slice_idx, :].T 
                    seg_slice = seg_data[:, slice_idx, :].T
                    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    
                    # Background
                    ax.imshow(img_slice, cmap='gray', origin='lower')
                    
                    # Overlay (Red) - Only shows if seg_data > 0
                    masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
                    ax.imshow(masked_seg, cmap='Reds', alpha=0.5, origin='lower')
                    
                    # Center Dot
                    ax.plot(center[0], center[2], 'bo', markersize=8, label='Center')
                    
                    # Crop Box
                    rect_x = roi_start[0]
                    rect_y = roi_start[2] 
                    rect_w = self.spatial_size[0]
                    rect_h = self.spatial_size[2]
                    
                    rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                           linewidth=2, edgecolor='lime', facecolor='none', label='Crop Box')
                    ax.add_patch(rect)
                    
                    ax.legend()
                    ax.set_title(f"Crop Validation: {Path(filename).name}\nSlice Y={slice_idx}")
                    ax.axis('off')
                    
                    plot_path = self.save_dir / f"{Path(filename).name}_validation.png"
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Visualization warning for {filename}: {e}")

            # --- 6. Handle Padding ---
            spatial_shape = np.array(image_tensor.shape[1:])
            
            pad_lower = np.maximum(-roi_start, 0)
            pad_upper = np.maximum(roi_end - spatial_shape, 0)
            
            current_data = image_tensor 
            
            if np.any(pad_lower > 0) or np.any(pad_upper > 0):
                pads = [(0, 0)] + list(zip(pad_lower, pad_upper))
                current_data_np = current_data.cpu().numpy()
                current_data_np = np.pad(current_data_np, pads, mode='constant', constant_values=1000)
                current_data = torch.from_numpy(current_data_np)
                roi_start = roi_start + pad_lower
            
            # --- 7. Apply Crop ---
            roi_end = roi_start + self.spatial_size
            roi_start_tuple = tuple(roi_start.astype(int).tolist())
            roi_end_tuple = tuple(roi_end.astype(int).tolist())

            cropper = SpatialCrop(roi_start=roi_start_tuple, roi_end=roi_end_tuple)
            d[key] = cropper(current_data)

        return d

class SimpleNiftiDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        image_paths = list(Path(data_dir).glob("*.nii.gz"))
        print(f"Found {len(image_paths)} .nii.gz files in {data_dir}")
        data = [
            {
                "image": str(image_path), 
                "report": "",
                "filename": str(image_path) 
            } 
            for image_path in image_paths
        ]
        super().__init__(data=data, transform=transform)
        
def get_args_parser():
    parser = argparse.ArgumentParser(description="Save embeddings from 3D NIfTI images using Spectre models")
    
    parser.add_argument("--data_dir", type=str, default=r"/home/20203686/own dataset", help="Directory to dataset")
    parser.add_argument("--save_dir", type=str, default=r"/home/20203686/crop embeds i e", help="Directory to save embeddings")
    parser.add_argument("--patch_size", type=int, nargs=3, default=(128, 128, 64), 
        help="Size of the 3D patches (H, W, D)",)
    parser.add_argument(
        "--model_variant", 
        type=str, 
        default="spectre-large-pretrained", 
        choices=MODEL_CONFIGS.keys(),
        help="Which Spectre model config to use"
    )
    parser.add_argument("--text_tokenizer", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    parser.add_argument("--lora_target_keywords", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--text_backbone_weights", type=str, default=None)
    parser.add_argument("--text_projection_weights", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--num_workers", type=int, default=0) 
    
    return parser
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    do_text_backbone = args.text_backbone_weights is not None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    val_dir = save_dir / "validation_imgs"
    val_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validation images will be saved to: {val_dir}")

    # --- CORRECT TRANSFORM PIPELINE ---
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.5, 0.5, 1.0), mode=("bilinear",)),
        CropLungsAlignedd(
            keys=["image"], 
            spatial_size=(512,512,384), 
            fast=True, 
            save_dir=val_dir
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        LargestMultipleCenterCropd(
            keys=("image",),
            patch_size=args.patch_size,
        ),
        GridPatchd(
            keys=("image",),
            patch_size=args.patch_size,
            overlap=0.0,
        ),
    ])

    print("Initializing Dataset...")
    dataset = SimpleNiftiDataset(args.data_dir, transform=transform)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=partial(
            extended_collate_siglip, 
            tokenizer=Qwen2TokenizerFast.from_pretrained(args.text_tokenizer) if do_text_backbone else None,
            tokenizer_max_length=4096,
            return_filenames=True,
        ),
    )

    print(f"Loading Spectre Image Model configuration: {args.model_variant}...")
    config = MODEL_CONFIGS[args.model_variant]
    
    image_model = SpectreImageFeatureExtractor.from_config(config)
    image_model.to(device).eval()

    print("\nStarting Inference...")
    
    for batch in tqdm(dataloader, desc="Processing batches", total=len(dataloader)):
        input_images = batch["image"].to(device) 
        
        filenames = [Path(f).name.split(".")[0] for f in batch["filename"]]
        save_paths = [save_dir / filename for filename in filenames]

        with torch.no_grad():
            image_features = image_model(input_images, grid_size=(4, 4, 6))
            save_embeddings(
                image_features, 
                [p / "spectre_embedding.npy" for p in save_paths]
            )

def save_embeddings(embeddings, save_paths):
    if isinstance(embeddings, torch.Tensor):
        embeddings = torch.split(embeddings, 1, dim=0)
        embeddings = [emb.squeeze(0) for emb in embeddings if emb.numel() > 0]
    
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