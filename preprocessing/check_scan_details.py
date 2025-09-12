import os
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


DATASET_PATH = Path("/gpfs/work5/0/tser0550/Datasets/CT-RATE/")


def main():
    # Get min, max and avg of the scan details
    ct_scans = list(DATASET_PATH.glob("dataset/*/*/*/*.nii.gz"))
    print(f"Num scans: {len(ct_scans)}")
    scan_details = []
    for scan in ct_scans:
        img = sitk.ReadImage(str(scan))
        spacing = img.GetSpacing()
        size = img.GetSize()
        scan_details.append(
            {
                "scan": str(scan),
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "size_x": size[0],
                "size_y": size[1],
                "size_z": size[2],
            }
        )
    df = pd.DataFrame(scan_details)
    # Add sizes when scan would be resampled to [0.75, 0.75, 1.5]
    df["size_x_resampled"] = df["size_x"] * (df["spacing_x"] / 0.75)
    df["size_y_resampled"] = df["size_y"] * (df["spacing_y"] / 0.75)
    df["size_z_resampled"] = df["size_z"] * (df["spacing_z"] / 1.5)

    print("Scan details:")
    print("\n")
    print(f"Num scans: {len(scan_details)}")
    print("\n")
    print(f"Min spacing: [{df['spacing_x'].min(), df['spacing_y'].min(), df['spacing_z'].min()}]")
    print(f"Max spacing: [{df['spacing_x'].max(), df['spacing_y'].max(), df['spacing_z'].max()}]")
    print(f"Avg spacing: [{df['spacing_x'].mean(), df['spacing_y'].mean(), df['spacing_z'].mean()}]")
    print(f"Med spacing: [{df['spacing_x'].median(), df['spacing_y'].median(), df['spacing_z'].median()}]")
    print("\n")
    print(f"Min size: [{df['size_x'].min(), df['size_y'].min(), df['size_z'].min()}]")
    print(f"Max size: [{df['size_x'].max(), df['size_y'].max(), df['size_z'].max()}]")
    print(f"Avg size: [{df['size_x'].mean(), df['size_y'].mean(), df['size_z'].mean()}]")
    print(f"Med size: [{df['size_x'].median(), df['size_y'].median(), df['size_z'].median()}]")
    print("\n")
    print("\n")
    print("Scan details (resampled to [0.75, 0.75, 1.5] spacing):")
    print("\n")
    print(f"min size: [{df['size_x_resampled'].min(), df['size_y_resampled'].min(), df['size_z_resampled'].min()}]")
    print(f"max size: [{df['size_x_resampled'].max(), df['size_y_resampled'].max(), df['size_z_resampled'].max()}]")
    print(f"avg size: [{df['size_x_resampled'].mean(), df['size_y_resampled'].mean(), df['size_z_resampled'].mean()}]")
    print(f"med size: [{df['size_x_resampled'].median(), df['size_y_resampled'].median(), df['size_z_resampled'].median()}]")
    
    df.to_csv("CT-RATE-details.csv", index=False)

if __name__ == "__main__":
    main()