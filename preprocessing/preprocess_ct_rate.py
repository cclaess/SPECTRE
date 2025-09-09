import os
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import SimpleITK as sitk


MANUALLY_CHECKED = {
    "train_1022_a": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "train_11401_a": ["1", "2", "3", "5", "6"],
    "train_1162_a": ["1", "2", "4", "5"],
    "train_13728_a": ["1", "2", "3", "5", "6", "7"],
    "train_13739_a": ["1", "2", "4", "5"],
    "train_17867_a": ["1", "2", "4", "5"],
    "train_18676_a": ["1", "2", "3", "5", "6"],
    "train_2871_a": ["1", "2", "3", "5"],
    "train_3821_a": ["1", "2", "4", "5"],
    "train_4288_a": ["1", "2", "3", "5"],
    "train_6331_a": ["1", "2", "4", "5"],
    "valid_1170_a": ["1", "2", "5"],
}


def main(input_dir: Path) -> None:
    """
    Main function to process the CT rate data.

    Args:
        input_dir (Path): Path to the input directory containing the CT-RATE data.
    """
    df = pd.concat([
        pd.read_csv(input_dir / "dataset" / "metadata" / "train_metadata.csv"),
        pd.read_csv(input_dir / "dataset" / "metadata" / "validation_metadata.csv"),
    ])
    scans = input_dir.glob("dataset/*/*/*")
    for scan in scans:
        reconstructions = sorted(scan.glob("*.nii.gz"), key=os.path.getsize, reverse=True)
        if len(reconstructions) == 0:
            print(f"No reconstructions found for {scan}")
            continue
        if scan.name in MANUALLY_CHECKED.keys():
            reconstructions_remove = [
                r for r in reconstructions if r.stem[:-4].split("_")[-1] in MANUALLY_CHECKED[scan.name]
            ]
            reconstructions_keep = [
                r for r in reconstructions if r.stem[:-4].split("_")[-1] not in MANUALLY_CHECKED[scan.name]
            ]
        else:
            # only keep the two largest files if more than 2 exist
            if len(reconstructions) > 2:
                reconstructions_keep = []

                # Check if any of the reconstructions has "thorax" in the description
                for r in reconstructions:
                    df_r = df.loc[df["VolumeName"] == r.name]
                    description = df_r["SeriesDescription"].values[0].lower()
                    if "thorax" in description or "lung" in description:
                        reconstructions_keep.append(r)
                
                # If reconstructions_keep < 2 and there's still reconstructions left, add up to 2 largest
                if len(reconstructions_keep) < 2:
                    reconstructions_keep.extend(reconstructions[:2 - len(reconstructions_keep)])

                # Remove the rest
                reconstructions_remove = [r for r in reconstructions if r not in reconstructions_keep]
            else:
                reconstructions_keep = list(reconstructions)
                reconstructions_remove = []
        
        for r in reconstructions_remove:
            # r.unlink()
            print(f"Removed {r}")
        for r in reconstructions_keep:
            df_r = df.loc[df["VolumeName"] == r.name]
            spacing_xy = df_r["XYSpacing"].values[0]
            spacing_xy = tuple([float(s) for s in spacing_xy[1:-1].split(",")])
            spacing_z = float(df_r["ZSpacing"].values[0])
            spacing = spacing_xy + (spacing_z,)
            origin = df_r["ImagePositionPatient"].values[0]
            origin = [float(o) for o in origin[1:-1].split(",")]
            origin = tuple(origin)
            rescale_intercept = int(df_r["RescaleIntercept"].values[0])
            img = sitk.ReadImage(str(r))
            img.SetSpacing(spacing)
            img.SetOrigin(origin)
            img = img + rescale_intercept
            sitk.WriteImage(img, str(r))

            
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to the input directory containing the data."
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    main(input_dir)