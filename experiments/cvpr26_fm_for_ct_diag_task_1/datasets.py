import os
import h5py

import torch
import pandas as pd
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(self, embeds_dir, csv_path, split, target_column=None):
        # Load CSV and filter by split
        df = pd.read_csv(csv_path)
        split_df = df[df['split'] == split].copy()

        # Build paths and label mapping
        self.paths = []
        self.label_mapping = {}

        for _, row in split_df.iterrows():
            # Extract filename without extension
            case_id = row['case_id']
            filename = case_id.split('.nii.gz')[0] if '.nii.gz' in case_id else case_id
            filename_base = filename.replace('.h5', '')  # Base filename for mapping

            # Construct path with .h5 extension
            h5_filename = filename_base + '.h5'
            path = os.path.join(embeds_dir, h5_filename)

            # Only add if file exists
            if os.path.exists(path):
                self.paths.append(path)
                self.label_mapping[filename_base] = int(row[target_column])
            else:
                print(f"Warning: File not found, skipping: {path}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        # Extract filename without extension from path
        filename = os.path.basename(path).replace('.h5', '')

        # Get label from CSV mapping
        if filename not in self.label_mapping:
            raise ValueError(f"Filename {filename} not found in label mapping")
        y = torch.tensor(self.label_mapping[filename]).long()

        # Load features from h5 file
        with h5py.File(path, 'r') as hf:
            y_hat = torch.tensor(hf['y_hat'][:]).float()

        return y_hat, y

    def _get_num_classes(self):
        # Get unique labels from the CSV mapping
        all_labels = set(self.label_mapping.values())
        return len(all_labels)
