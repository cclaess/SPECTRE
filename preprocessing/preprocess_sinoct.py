import shutil
from pathlib import Path
from argparse import ArgumentParser

import pydicom
import SimpleITK as sitk


def load_dicom_series(series_path: Path) -> sitk.Image:
    """
    Load a DICOM series from the given path.

    Args:
        series_path (Path): Path to the DICOM series directory.

    Returns:
        sitk.Image: Loaded DICOM image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_path))
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def get_slice_thickness(series_path: Path) -> float:
    """
    Get the slice thickness from the DICOM series.
    SimpleITK does not read them correctly from these scans, pydicom does

    Args:
        series_path (Path): Path to the DICOM series directory.

    Returns:
        float: Slice thickness.
    """
    dicom_files = sorted(series_path.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_path}")
    
    ds = pydicom.dcmread(dicom_files[len(dicom_files) // 2])  # Read the middle file for slice thickness
    if 'SliceThickness' in ds:
        return float(ds.SliceThickness)
    else:
        raise ValueError(f"Slice thickness not found in DICOM files at {series_path}")


def main(input_dir: Path) -> None:
    """
    Main function to process the SinoCT data.

    Args:
        input_dir (Path): Path to the input directory containing the SinoCT data.
    """
    patient_dirs = sorted(input_dir.glob("batch_*/series_*"))
    for patient_dir in patient_dirs:
        image = load_dicom_series(patient_dir / "reconstructed_image")
        spacing = image.GetSpacing()
        image.SetSpacing((spacing[0], spacing[1], get_slice_thickness(patient_dir / "reconstructed_image")))  # Adjust Z-spacing
        output_path = patient_dir / "reconstructed_image.nii.gz"
        sitk.WriteImage(image, str(output_path))
        print(f"Processed {patient_dir.name}: saved to {output_path}")
        shutil.rmtree(patient_dir / "reconstructed_image")  # Remove the original DICOM series directory
        shutil.rmtree(patient_dir / "sinogram")  # Remove the sinogram directory

            
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default="./", help="Path to the input directory containing the data."
    )
    args = parser.parse_args()
    input_dir = Path(r"E:\Datasets\SinoCT\ctsinogram\head_ct_dataset_anon")

    main(input_dir)