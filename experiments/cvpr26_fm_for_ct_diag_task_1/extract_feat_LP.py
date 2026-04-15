import os
import h5py
import argparse
import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from monai.data import Dataset, ThreadDataLoader
from monai.transforms import (
    MapTransform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    CopyItemsd,
    Spacingd,
    ResizeWithPadOrCropd,
    DeleteItemsd,
    ToTensord,
)

from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", "--imgs_path", dest="imgs_path", type=str,
                        default='/workspace/inputs',
                        help='Path to input images directory')
    parser.add_argument("-o", "--output", "--dest", dest="dest", type=str,
                        default='/workspace/outputs',
                        help='Destination folder to save features')
    parser.add_argument("--masks_path", type=str, default=None,
                        help='Path to foreground masks for roi-disease (defaults to None)')
    parser.add_argument("--checkpoint_backbone", type=str, 
                        default='/app/weights/backbone_weights.pt',
                        help='Path to backbone model checkpoint')
    parser.add_argument("--checkpoint_combiner", type=str, 
                        default='/app/weights/combiner_weights.pt',
                        help='Path to feature combiner model checkpoint')
    parser.add_argument("--patch_size", type=int, nargs=3, default=(128, 128, 64), 
                        help='Patch size for feature extraction')
    parser.add_argument("--batch_size", type=int, default=1, 
                        help='Batch size for feature extraction')
    parser.add_argument("--dump_dir", type=str, default=None, 
                        help='Directory to save debug images and masks')
    parser.add_argument("--num_workers", type=int, default=0, 
                        help='Number of workers for data loading')

    return parser.parse_args()


class MaskCenterCropd(MapTransform):
    """Custom MONAI transform to crop around mask center with padding."""
    def __init__(
        self, 
        keys, 
        mask_key="mask", 
        roi_size=(384, 384, 192), 
        fg_labels=None
    ):
        super().__init__(keys)
        self.mask_key = mask_key
        self.roi_size = roi_size
        self.fg_labels = fg_labels
        self.img_key = 'image'
      
    def __call__(self, data):
        d = dict(data)  
          
        # Get mask center  
        mask_arr = d[self.mask_key]  
        if len(mask_arr.shape) == 4:  # Remove channel dimension if present  
            mask_arr = mask_arr[0]  
        
        # make binary mask for specified foreground labels
        if self.fg_labels is not None:
            mask_arr = np.isin(mask_arr, self.fg_labels).astype(np.uint8)

            coords = np.argwhere(mask_arr == 1)  
            if coords.size == 0:  
                # fall back to original mask to get coordinates
                mask_arr_orig = d['mask_original']
                if len(mask_arr_orig.shape) == 4:  # Remove channel dimension if present  
                    mask_arr_orig = mask_arr_orig[0]
                mask_arr_orig = np.isin(mask_arr_orig, self.fg_labels).astype(np.uint8)
                shape_ori = mask_arr_orig.shape
                shape_resampled = mask_arr.shape
                coords = np.argwhere(mask_arr_orig == 1)
                # if no coordinates found, just use center of image
                if coords.size == 0:
                    coords = np.array([[shape_resampled[0]//2, shape_resampled[1]//2, shape_resampled[2]//2]])
                else:
                    # scale the coordinates to resampled shape
                    scale_z = shape_resampled[0] / shape_ori[0]
                    scale_y = shape_resampled[1] / shape_ori[1]
                    scale_x = shape_resampled[2] / shape_ori[2]
                    coords = np.array([[int(c[0]*scale_z), int(c[1]*scale_y), int(c[2]*scale_x)] for c in coords])
            center = tuple(coords.mean(axis=0).astype(int))  

        else:
            # center cropping if no fg_labels provided
            img_arr = d[self.img_key]
            shape_img = img_arr.shape[1:] if len(img_arr.shape) == 4 else img_arr.shape
            center = (shape_img[0]//2, shape_img[1]//2, shape_img[2]//2)
            
        # Crop each key around the mask center  
        for key in self.keys:  
            arr = d[key]  
            has_channel = len(arr.shape) == 4  
            if has_channel:  
                arr_data = arr[0]  # Remove channel for processing  
            else:  
                arr_data = arr  
            cropped = self._crop_with_padding(arr_data, center, self.roi_size)  
              
            if has_channel:  
                d[key] = cropped[np.newaxis, ...]  # Add channel back  
            else:  
                d[key] = cropped  

        return d  
      
    def _crop_with_padding(self, arr, center, size):  
        """Crop 3D array with zero padding around center (z, y, x)."""  
        zc, yc, xc = center  
        dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2  
          
        z_start, z_end = zc - dz, zc + dz  
        y_start, y_end = yc - dy, yc + dy  
        x_start, x_end = xc - dx, xc + dx  
          
        if torch.is_tensor(arr):
            cropped = torch.zeros(size, dtype=arr.dtype, device=arr.device)
        else:
            cropped = np.zeros(size, dtype=arr.dtype)  
          
        z_start_valid = max(z_start, 0)  
        y_start_valid = max(y_start, 0)  
        x_start_valid = max(x_start, 0)  
          
        z_end_valid = min(z_end, arr.shape[0])  
        y_end_valid = min(y_end, arr.shape[1])  
        x_end_valid = min(x_end, arr.shape[2])  
          
        z_off = z_start_valid - z_start  
        y_off = y_start_valid - y_start  
        x_off = x_start_valid - x_start  
          
        cropped[  
            z_off:z_off + (z_end_valid - z_start_valid),  
            y_off:y_off + (y_end_valid - y_start_valid),  
            x_off:x_off + (x_end_valid - x_start_valid)  
        ] = arr[  
            z_start_valid:z_end_valid,  
            y_start_valid:y_end_valid,  
            x_start_valid:x_end_valid  
        ]  
          
        return cropped  


def get_image_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-1000, 
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.5, 0.5, 1.0), mode='bilinear'),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=[512, 512, 256]),
        ToTensord(keys=["image"]),
    ])


def get_image_mask_transforms():
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-1000, 
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        CopyItemsd(keys=["mask"], names=["mask_original"]),  # Needed by MaskCenterCropd
        Spacingd(
            keys=["image", "mask"], 
            pixdim=(0.5, 0.5, 1.0), 
            mode=['bilinear', 'nearest']
        ),
        MaskCenterCropd(
            keys=["image", "mask"], 
            mask_key="mask", 
            roi_size=(384, 394, 192), 
            fg_labels=[1]
        ),
        ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=[384, 384, 192]),
        DeleteItemsd(keys=["mask_original"]),
        ToTensord(keys=["image", "mask"]),
    ])


def main(args):

    config = MODEL_CONFIGS['spectre-large-pretrained']
    config['backbone_checkpoint_path_or_url'] = args.checkpoint_backbone
    config['feature_combiner_checkpoint_path_or_url'] = args.checkpoint_combiner

    model = SpectreImageFeatureExtractor.from_config(config)
    model.eval()
    model.to(device)

    imgs_path = args.imgs_path

    # Process all image files in the input directory
    datalist = []
    imgs_files = sorted([f for f in os.listdir(imgs_path) if f.endswith('.nii.gz')])
    if args.masks_path:
        imgs_files = [f for f in imgs_files if os.path.exists(os.path.join(args.masks_path, f))]

    for img_file in imgs_files:
        img_id = img_file.split('.nii.gz')[0]
        img_full_path = os.path.join(imgs_path, img_file)

        # Check if h5 file already exists for this image, skip if it does
        out_path = os.path.join(args.dest, f'{img_id}.h5')
        if os.path.exists(out_path):
            print(f'Output already exists for {img_id}, skipping...')
            continue
        # Construct path to foreground mask
        mask_full_path = os.path.join(args.masks_path, img_file) if args.masks_path is not None else None
        assert os.path.exists(img_full_path), f'Image file not found: {img_full_path}'
        if mask_full_path is not None:
            assert os.path.exists(mask_full_path), f'Mask file not found: {mask_full_path}'
            datalist.append({
                "image": img_full_path,
                "mask": mask_full_path,
                'filename': img_id
            })
        else:
            datalist.append({
                "image": img_full_path,
                'filename': img_id
            })

    if args.masks_path is None:
        image_transforms = get_image_transforms()
    else:
        image_transforms = get_image_mask_transforms()


    # Create dataloader
    dataset = Dataset(data=datalist, transform=image_transforms)
    dataloader = ThreadDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Extract embeddings and save immediately after each batch
    processed_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            # Handle both dict and list outputs from DataLoader
            if isinstance(batch, dict):
                images = batch["image"]
                masks = batch.get("mask", None)
                filenames = batch.get(
                    'filename', 
                    [f'batch_{batch_idx}_sample_{i}' for i in range(images.shape[0])],
                )
            else:
                raise ValueError(
                    f"""Expected batch to be a dict, but got {type(batch)}. Please ensure that the 
                    DataLoader returns a dictionary with keys `image`, `mask`, and `filename`."""
                )

            if not isinstance(filenames, (list, tuple)):
                raise ValueError(
                    f"""Expected 'filename' to be a list, but got {type(filenames)}. Please ensure 
                    that the DataLoader returns a list of filenames for each batch."""
                )
            assert len(filenames) == images.shape[0], \
                f'Number of filenames {len(filenames)} does not match batch size {images.shape[0]}'

            if args.dump_dir:
                for i, filename in enumerate(filenames):
                    image_np = images[i].cpu().numpy()

                    if image_np.shape[0] == 1:
                        image_np = image_np[0]

                    # Convert to SimpleITK image and save
                    image_sitk = sitk.GetImageFromArray(image_np)
                    image_sitk.SetSpacing((0.5, 0.5, 1.0))
                    image_output_path = os.path.join(args.dump_dir, f'{filename}_image.nii.gz')
                    sitk.WriteImage(image_sitk, image_output_path)

                    # Save mask if available
                    if masks is not None:
                        mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]

                        # Remove channel dimension if it's 1
                        if mask_np.shape[0] == 1:
                            mask_np = mask_np[0]

                        # Convert to SimpleITK image and save
                        mask_sitk = sitk.GetImageFromArray(mask_np)
                        mask_sitk.SetSpacing((0.5, 0.5, 1.0))
                        mask_output_path = os.path.join(args.dump_dir, f'{filename}_mask.nii.gz')
                        sitk.WriteImage(mask_sitk, mask_output_path)

            # Move to device
            images = images.to(device, non_blocking=True)

            B, C, H, W, D = images.shape
            pH, pW, pD = args.patch_size

            images = images.view(
                B, C,
                H // pH, pH,
                W // pW, pW,
                D // pD, pD,
            ).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, -1, C, pH, pW, pD).contiguous()

            # Forward pass through the model to get image latent features
            outputs = model(
                images,
                grid_size=(
                    H // pH,
                    W // pW,
                    D // pD,
                )
            )

            # Pool and flatten embeddings
            image_embeddings = outputs[:, 0, :].detach().cpu()  # (B, feature_dim)

            # Save h5 files immediately for this batch
            for i, filename in enumerate(filenames):
                single_out_path = os.path.join(args.dest, f'{filename}.h5')

                # Save in h5 format
                with h5py.File(single_out_path, 'w') as hf:
                    hf.create_dataset('y_hat', data=image_embeddings[i].numpy())

                processed_count += 1

            # Clean up memory immediately after saving
            del outputs, images, image_embeddings
            torch.cuda.empty_cache()

  
if __name__ == "__main__":

    args = get_args()

    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok=True)
    os.makedirs(args.dest, exist_ok=True)

    main(args)
