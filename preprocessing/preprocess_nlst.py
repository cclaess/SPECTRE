from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from itertools import combinations
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

def longest_common_prefix_length(s1, s2):
    """Returns the length of the longest common prefix of two strings."""
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return i

def compute_lcp_distance_matrix(strings):
    """Computes a distance matrix using LCP similarity."""
    n = len(strings)
    matrix = np.zeros((n, n))
    
    for i, j in combinations(range(n), 2):
        lcp_length = longest_common_prefix_length(strings[i], strings[j])
        max_length = max(len(strings[i]), len(strings[j]))
        distance = 1 - (lcp_length / max_length)  # Normalize to [0,1] range
        
        matrix[i, j] = distance
        matrix[j, i] = distance  # Symmetric matrix
    
    return matrix

def divide_into_groups(strings):
    if len(strings) <= 1:
        return strings, []  # If only one string, return it as the first group
    
    distance_matrix = compute_lcp_distance_matrix(strings)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Split into 2 clusters
    labels = fcluster(linkage_matrix, 2, criterion='maxclust')
    
    # Divide strings based on cluster labels
    group1 = [strings[i] for i in range(len(strings)) if labels[i] == 1]
    group2 = [strings[i] for i in range(len(strings)) if labels[i] == 2]
    
    return group1, group2


DATA_DIR = r"/gpfs/work4/0/tese0618/Datasets/NLST"
CSV_PATH = r"/gpfs/work4/0/tese0618/Datasets/NLST/package-nlst-780.2021-05-28/nlst_780/nlst780.idc.delivery.052821" \
           r"/nlst_780_screen_idc_20210527.csv/nlst_780_screen_idc_20210527.csv"

RECON_FILTERS = {
    1: "GE Bone",
    2: "GE Standard",
    3: "GE, other",
    4: "Philips D",
    5: "Philips C",
    6: "Philips, other",
    7: "Siemens B50F",
    8: "Siemens B30",
    9: "Siemens, other",
    10: "Toshiba FC10",
    11: "Toshiba FC51",
    12: "Toshiba, other",
}


def main():
    df = pd.read_csv(CSV_PATH)
    # df_new = pd.DataFrame(columns=["pid", "study_yr", "ct_image1", "ct_image2"])

    ct_scan_folders = sorted(list(Path(DATA_DIR).glob("*/*")))
    for folder in ct_scan_folders:
        if not folder.is_dir():
            continue
        
        pid = folder.parent.name
        study_yr = 0 if "1999" in folder.name else 1 if "2000" in folder.name else 2
        try:
            num_filters = 1 if np.isnan(
                df[(df["pid"] == int(pid)) & (df["study_yr"] == int(study_yr))]["ct_recon_filter2"].values[0]
            ) else 2
        except Exception as e:
            print(f"Error for {pid}: ", e)
            num_filters = 1
       
        nii_files = list(folder.glob("*.nii.gz"))
        # Find scans with the same reconstruction filter based on their name
        nii_names = [nii_file.name for nii_file in nii_files]
        nii_names_clean = ["-".join(nii_name.split("-")[1:-1]) for nii_name in nii_names]

        # Filter out duplicate names
        nii_names_clean_unique = list(set(nii_names_clean))
        
        images_keep = []
        # Check if there is only one or none .nii.gz file in the folder
        if len(nii_files) == 0:
            print(f"No .nii.gz files found in {folder}.")
            continue
        elif len(nii_files) == 1:
            # Cannot be same reconstructions with different slice thicknesses
            images_keep.append(nii_files[0])

        
        elif len(nii_files) == 2 and num_filters == 2:
            # Check if there's two kernels in the df, if so, keep both
            # ct_recon_filter2 should contain a number
            images_keep.append(nii_files[0])
            images_keep.append(nii_files[1])
        
        # Now we will have to remove one or more files, depending on one or two different reconstructions
        # Check how many kernels are in the df
        elif len(nii_names_clean_unique) <= 2:
            # Some not corect in csv
            for nii_name_clean_unique in nii_names_clean_unique:
                # Get the files that match the unique name
                matching_files = [nii_file for nii_file in nii_files if nii_name_clean_unique in nii_file.name]
                if len(matching_files) == 1:
                    images_keep.append(matching_files[0])
                else:
                    # Get the file with the most slices
                    biggest_file = max(matching_files, key=lambda x: sitk.ReadImage(str(x)).GetSize()[2])
                    images_keep.append(biggest_file)
        else:
            # Aparently some reconstructions with the same filter have different names
            # We have to find out which ones have the same filter and remove all but one of them

            groups = divide_into_groups(nii_names_clean_unique)
            for group in groups:
                matching_files = []
                for nii_name_clean_unique in group:
                    matching_files.extend([nii_file for nii_file in nii_files if nii_name_clean_unique in nii_file.name])
                if len(matching_files) == 0:
                    continue
                elif len(matching_files) == 1:
                    images_keep.append(matching_files[0])
                else:
                    # Get the file with the most slices
                    biggest_file = max(matching_files, key=lambda x: sitk.ReadImage(str(x)).GetSize()[2])
                    images_keep.append(biggest_file)
            
        # df_new = pd.concat([df_new, pd.DataFrame.from_records([{
        #     "pid": pid,
        #     "study_yr": study_yr,
        #     "ct_image1": str(images_keep[0]) if len(images_keep) > 0 else None,
        #     "ct_image2": str(images_keep[1]) if len(images_keep) > 1 else None,
        # }])])
        images_remove = []
        for nii_file in nii_files:
            if nii_file not in images_keep:
                images_remove.append(nii_file)
        for nii_file in images_remove:
            nii_file.unlink()
            print(nii_file)
        # Remove the images from the folder
    
    
    # df_new.to_csv("/gpfs/work4/0/tese0618/nlst_test.csv", index=False)
    

if __name__ == "__main__":
  main()
