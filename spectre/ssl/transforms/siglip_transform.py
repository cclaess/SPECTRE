from typing import Tuple

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    CastToTyped,
    MapTransform,
    RandomizableTransform,
)
from monai.config import KeysCollection

from spectre.transforms import SWSpatialCropSamplesd


class SigLIPTransform(Compose):
    def __init__(
        self, 
        input_size: Tuple[int, int, int] = (128, 128, 64),
        dtype: str = "float32",
    ):
        assert dtype in ["float16", "float32"], "dtype must be either 'float16' or 'float32'"
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
                CastToTyped(keys=("image",), dtype=getattr(torch, dtype)),
                
                # Crop the volume into equal non-overlapping samples
                SWSpatialCropSamplesd(
                    keys=("image",),
                    patch_size=input_size,
                    overlap=0.0,
                ),

                # load the text data
                GenerateReportTransform(
                    keys=("findings", "impressions", "icd10"), 
                    icd10_range_lower=0.1, 
                    likelyhood_original=0.5
                ),
            ]
        )


class GenerateReportTransform(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        max_num_icd10=35,
        likelyhood_original=0.5,
        drop_chance=0.3,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys (KeysCollection): Keys to be processed.
            max_num_icd10 (float): Maximum number of ICD10 codes to include in the report.
            likelyhood_original (float): Likelihood weight for selecting the first element.
            drop_chance (float): Probability of dropping the findings and icd10 if present.
            allow_missing_keys (bool): Whether to allow missing keys in the data.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.max_num_icd10 = max_num_icd10
        self.likelyhood_original = likelyhood_original
        
        RandomizableTransform.__init__(self, max_num_icd10, likelyhood_original)
        
    def __call__(self, data):
        # Expect data to be a dict with keys: 'findings', 'impressions', 'icd10'
        findings = data.get("findings", [])
        impressions = data.get("impressions", [])
        icd10_codes = data.get("icd10", [])
        
        # If ICD10 codes come as a string, convert them to a list.
        if isinstance(icd10_codes, str):
            icd10_codes = icd10_codes.split(";")
        # if no icd10 codes are present, set it to an empty list
        if not icd10_codes or torch.isnan(icd10_codes):
            icd10_codes = []
        
        report = ""
        
        # Randomly drop findings and icd10 codes based on the drop chance.
        if self.R.random() < self.drop_chance:
            findings = []
        if self.R.random() < self.drop_chance:
            icd10_codes = []

        # Randomly select a finding and impression using weighted probabilities.
        if len(findings) > 0:
            num_elements = len(findings)
            weights = [self.likelyhood_original] + [(1 - self.likelyhood_original) / (num_elements - 1)] * (num_elements - 1)
            selected_finding = self.R.choice(findings, p=weights)
        
            # Add finding to the report.
            report += f"Findings: {selected_finding}\n"
            # Remove the word "Impressions" from the selected finding.
            selected_finding = selected_finding.replace("Impressions", "").replace("impressions", "")

        if len(impressions) > 0:
            num_elements = len(impressions)
            weights = [self.likelyhood_original] + [(1 - self.likelyhood_original) / (num_elements - 1)] * (num_elements - 1)
            # Select an impression based on the weights.
            selected_impression = self.R.choice(impressions, p=weights)

            # Add impression to the report.
            report += f"Impressions: {selected_impression}\n"

        # check if icd10_codes is empty or not
        if len(icd10_codes) > 0:
                
            # Randomly      
            if self.max_num_icd10 > 0:
                # Ensure we do not exceed the available number of codes.
                num_codes = min(self.max_num_icd10, len(icd10_codes))
                # Sample a subset of ICD10 codes without replacement.
                selected_icd10 = self.R.choice(icd10_codes, size=num_codes, replace=False).tolist()
            
            # Add ICD10 codes to the report.
            report += f"ICD10: {', '.join(selected_icd10)}\n"
        
        data["report"] = report
        return data
