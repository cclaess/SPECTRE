from typing import Tuple

import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    CastToTyped,
    RandSpatialCropd,
    MapTransform,
    Randomizable,
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
        global_size = (
            384 + input_size[0],
            384 + input_size[1],
            256 + input_size[2],
        )
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
                ResizeWithPadOrCropd(keys=("image",), spatial_size=global_size),
                CastToTyped(keys=("image",), dtype=getattr(torch, dtype)),
                
                # Crop the volume into equal non-overlapping samples
                RandSpatialCropd(
                    keys=("image",),
                    roi_size=(384, 384, 256),
                    random_size=False,
                    random_center=True,
                ),
                SWSpatialCropSamplesd(
                    keys=("image",),
                    patch_size=input_size,
                    overlap=0.0,
                ),

                # load the text data
                GenerateReportTransform(
                    keys=("findings", "impressions", "icd10"), 
                    max_num_icd10=20, 
                    likelyhood_original=0.5,
                    drop_chance=0.3,
                ),
            ]
        )


class GenerateReportTransform(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        max_num_icd10=20,
        likelyhood_original=0.5,
        drop_chance=0.3,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.max_num_icd10 = max_num_icd10
        self.likelyhood_original = likelyhood_original
        self.drop_chance = drop_chance

        # Random states (purely indices/flags)
        self.drop_findings = False
        self.drop_icd10 = False
        self.finding_idx = None
        self.impression_idx = None
        self.icd10_indices = []

    def randomize(self, data):
        findings = data.get("findings", [])
        impressions = data.get("impressions", [])
        icd10_codes = data.get("icd10", [])

        if isinstance(icd10_codes, str):
            icd10_codes = icd10_codes.split(";")
        if not isinstance(icd10_codes, list):
            icd10_codes = []

        self.drop_findings = self.R.random() < self.drop_chance
        self.drop_icd10 = self.R.random() < self.drop_chance
        self.finding_idx = None
        self.impression_idx = None
        self.icd10_indices = []

        if not self.drop_findings and findings:
            num_elements = len(findings)
            if num_elements == 1:
                self.finding_idx = 0
            else:
                weights = [self.likelyhood_original] + [(1 - self.likelyhood_original) / (num_elements - 1)] * (num_elements - 1)
                self.finding_idx = int(self.R.choice(np.arange(num_elements), p=weights))

        if impressions:
            num_elements = len(impressions)
            if num_elements == 1:
                self.impression_idx = 0
            else:
                weights = [self.likelyhood_original] + [(1 - self.likelyhood_original) / (num_elements - 1)] * (num_elements - 1)
                self.impression_idx = int(self.R.choice(np.arange(num_elements), p=weights))

        if not self.drop_icd10 and icd10_codes:
            num_codes = min(self.max_num_icd10, len(icd10_codes))
            self.icd10_indices = self.R.choice(len(icd10_codes), size=num_codes, replace=False).tolist()

    def __call__(self, data):
        self.randomize(data)

        findings = data.get("findings", [])
        impressions = data.get("impressions", [])
        icd10_codes = data.get("icd10", [])

        if isinstance(icd10_codes, str):
            icd10_codes = icd10_codes.split(";")
        if not isinstance(icd10_codes, list):
            icd10_codes = []

        report = ""

        if self.finding_idx is not None and self.finding_idx < len(findings):
            finding = findings[self.finding_idx].replace("Impressions", "").replace("impressions", "")
            report += f"Findings: {finding}\n"

        if self.impression_idx is not None and self.impression_idx < len(impressions):
            impression = impressions[self.impression_idx]
            report += f"Impressions: {impression}\n"

        if self.icd10_indices:
            selected_icd10 = [icd10_codes[i] for i in self.icd10_indices if i < len(icd10_codes)]
            report += f"ICD10: {'; '.join(selected_icd10)}\n"

        data["report"] = report
        return data
