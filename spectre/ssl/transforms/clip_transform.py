from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
)

from spectre.transforms import SWSpatialCropSamplesd
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms import Transform
from monai.config import DtypeLike, KeysCollection, SequenceStr

from transformers import AutoTokenizer
class CLIPTransform(Compose):
    def __init__(
            self, 
            input_size: Tuple[int, int, int] = (128, 128, 64),
        ):
        super().__init__(
            [
                LoadImaged(keys=("image",)),
                EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=("image",), 
                    a_min=-1000, 
                    a_max=1000, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True
                ),
                Orientationd(keys=("image",), axcodes="RAS"),
                Spacingd(keys=("image",), pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
                ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, 256)),
                
                # Crop the volume into equal non-overlapping samples
                SWSpatialCropSamplesd(
                    keys=("image",),
                    patch_size=input_size,
                    overlap=0.0,
                )
            ]
        )


class SigLipTransform(Compose):
    def __init__(
            self, 
            input_size: Tuple[int, int, int] = (128, 128, 64),
        ):
        super().__init__(
            [
                LoadImaged(keys=("image",)),
                EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=("image",), 
                    a_min=-1000, 
                    a_max=1000, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True
                ),
                Orientationd(keys=("image",), axcodes="RAS"),
                Spacingd(keys=("image",), pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
                ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, 256)),
                
                # Crop the volume into equal non-overlapping samples
                SWSpatialCropSamplesd(
                    keys=("image",),
                    patch_size=input_size,
                    overlap=0.0,
                ),

                # load the text data
                GenerateReportTransform(keys=("findings", "impressions", "icd10"), icd10_range_lower=0.1, likelyhood_original=0.5),
                TokenizeTransform(keys=("image", "report"))

            ]
        )


class GenerateReportTransform(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        icd10_range_lower=1.0,
        likelyhood_original=0.5,
        allow_missing_keys: bool = False
    ):
        """
        Args:
            keys (KeysCollection): Keys to be processed.
            icd10_range_lower (float): A value between 0 and 1 representing the lower bound 
                                       for the percentage of ICD10 codes to include.
            likelyhood_original (float): Likelihood weight for selecting the first element.
            allow_missing_keys (bool): Whether to allow missing keys in the data.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.icd10_range_lower = icd10_range_lower
        self.likelyhood_original = likelyhood_original
        
        RandomizableTransform.__init__(self, icd10_range_lower, likelyhood_original)
        
    def __call__(self, data):
        # Expect data to be a dict with keys: 'findings', 'impressions', 'icd10'
        findings = data.get("findings", [])
        impressions = data.get("impressions", [])
        icd10_codes = data.get("icd10", [])
        
        # If ICD10 codes come as a string, convert them to a list.
        if isinstance(icd10_codes, str):
            icd10_codes = icd10_codes.split(",")
        
        # Ensure that we have at least three findings and impressions.
        if len(findings) < 3:
            raise ValueError("Expected at least 3 findings")
        if len(impressions) < 3:
            raise ValueError("Expected at least 3 impressions")
        
        # Define weights for selecting the first 3 items.
        weights = [
            self.likelyhood_original,
            (1 - self.likelyhood_original) / 2,
            (1 - self.likelyhood_original) / 2,
        ]
        
        # Randomly select a finding and impression using weighted probabilities.
        selected_finding = self.R.choice(findings[:3], p=weights)
        selected_impression = self.R.choice(impressions[:3], p=weights)
        
        # Draw a value between icd10_range_lower and 1.0 to determine the percentage of ICD10 codes to include.
        icd10_percentage = self.R.uniform(self.icd10_range_lower, 1.0)
        num_codes = int(len(icd10_codes) * icd10_percentage)
        
        if num_codes > 0:
            # Ensure we do not exceed the available number of codes.
            num_codes = min(num_codes, len(icd10_codes))
            # Sample a subset of ICD10 codes without replacement.
            selected_icd10 = self.R.choice(icd10_codes, size=num_codes, replace=False).tolist()
        else:
            selected_icd10 = []
        
        # Construct the final report string.
        report = (
            f"Findings: {selected_finding}\n"
            f"Impressions: {selected_impression}\n"
            f"ICD10: {', '.join(selected_icd10) if selected_icd10 else ''}"
        )
        
        data["report"] = report
        return data
    
class TokenizeTransform(MapTransform):
    
    def __init__(self, keys: KeysCollection, tokenizer_name=None, text_key="report"):

        self.keys = keys
        if tokenizer_name is None:
            tokenizer_name = "infgrad/jasper_en_vision_language_v1"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.text_key = text_key 
    def __call__(self, data):
        
        tokenizer_output = self.tokenizer(str(data[self.text_key]), add_special_tokens=True)
        data["input_ids"] = tokenizer_output["input_ids"]
        data["attention_mask"] = tokenizer_output["attention_mask"]
        return data




if __name__ == "__main__":

    # Save some example data after transforming it.
    import os
    import SimpleITK as sitk

    data = {"image": r"data/test_data/train_1_a_1.nii.gz"}
    transform = CLIPTransform()
    transformed_data = transform(data)

    # Save the different crops to a folder for visualization.
    output_dir = r"data/test_data/clip_transform_output"
    os.makedirs(output_dir, exist_ok=True)

    for i, patch in enumerate(transformed_data):

        # Save the crops
        patch_img = sitk.GetImageFromArray(patch["image"].squeeze(0).numpy())
        patch_img.SetSpacing((1.5, 0.75, 0.75))
        patch_path = os.path.join(output_dir, f"{i}_crop.nii.gz")
        sitk.WriteImage(patch_img, patch_path)

