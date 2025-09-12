import os
import shutil
import pandas as pd
import numpy as np
import re
from pathlib import Path

def extract_scan_id(volume_name):
    return re.sub(r'_\d+\.nii\.gz$', '', volume_name)

def process_dataset(base_path):
    base_path = Path(base_path)
    temp_folder = base_path / 'CT-RATE' / 'temp'
    dataset_folder = base_path / 'CT-RATE' / 'dataset'
    csv_path = temp_folder / 'CT-RATE-metadata-enhanced.csv'

    # Read CSV
    df = pd.read_csv(csv_path)

    # Extract scan_id
    df['scan_id'] = df['VolumeName'].apply(extract_scan_id)

    # Compute upper bound using Tukey rule
    skull = df['skull_percentage']
    q1 = skull.quantile(0.25)
    q3 = skull.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    # Check file existence in temp folder
    df['temp_path'] = df['VolumeName'].apply(lambda x: temp_folder / x)
    df['file_exists'] = df['temp_path'].apply(lambda p: p.is_file())

    # Determine upper outliers (but preserve NaNs as non-outliers)
    df['is_upper_outlier'] = df['skull_percentage'] > upper_bound

    # Initial exclusion: missing file or upper outlier
    df['exclude'] = ((~df['file_exists']) | df['is_upper_outlier']).astype(int)

    # Override exclusion if all reconstructions of a scan would be excluded for being outliers
    for scan_id, group in df.groupby('scan_id'):
        excluded = group['exclude']
        missing = ~group['file_exists']
        outliers = group['is_upper_outlier']
        # Check if ALL exclusions are due to outliers (and not due to missing files)
        if all(outliers & ~missing):
            df.loc[group.index, 'exclude'] = 0  # Include them

    # Process files: delete excluded, replace included
    for _, row in df.iterrows():
        volname = row['VolumeName']
        is_excluded = row['exclude'] == 1
        source_path = temp_folder / volname

        # Decide destination based on VolumeName prefix
        if volname.startswith('train_'):
            dest_root = dataset_folder / 'train'
        elif volname.startswith('valid_'):
            dest_root = dataset_folder / 'valid'
        else:
            continue  # Unknown case, skip

        # Build expected dataset path based on folder structure
        parts = volname.replace('.nii.gz', '').split('_')
        subfolder = '_'.join(parts[:2])
        subsubfolder = '_'.join(parts[:3])
        dest_path = dest_root / subfolder / subsubfolder / volname

        # Delete excluded files from dataset
        if is_excluded and dest_path.exists():
            print(f"Deleting excluded file: {dest_path}")
            dest_path.unlink()

        # Copy included files to dataset (overwrite)
        if not is_excluded and source_path.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Copying included file to dataset: {dest_path}")
            shutil.copy2(source_path, dest_path)

    # Save updated DataFrame with exclusion column
    updated_csv_path = temp_folder / 'CT-RATE-metadata-enhanced-with-exclusions.csv'
    df.to_csv(updated_csv_path, index=False)
    print(f"\nUpdated metadata saved to: {updated_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process CT-RATE dataset exclusions/inclusions.")
    parser.add_argument("path", type=str, help="Path to the dataset base directory (containing CT-RATE)")
    args = parser.parse_args()

    process_dataset(args.path)
    print("Processing completed.")
