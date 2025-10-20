from copy import deepcopy
from typing import Any, Mapping, Hashable

from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable


class RandomReportTransformd(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        max_num_icd10=20,
        keep_original_prob=0.5,
        drop_prob=0.3,
        allow_missing_keys: bool = False,
    ):
        assert all(str(key) in ["findings", "impressions", "icd10"] for key in keys), \
            "keys must be one of ['findings', 'impressions', 'icd10']"
        
        super().__init__(keys, allow_missing_keys)
        self.max_num_icd10 = max_num_icd10
        self.keep_original_prob = keep_original_prob
        self.drop_prob = drop_prob

        self._rand_state = {}
    
    def randomize(self, data: Any = None) -> None:
        self._rand_state.clear()

        for key in self.keys:
            if str(key) == "findings":
                self._rand_state["drop_findings"] = self.R.random() < self.drop_prob
                self._rand_state["keep_findings_original"] = self.R.random() < self.keep_original_prob
            
            elif str(key) == "impressions":
                self._rand_state["keep_impressions_original"] = self.R.random() < self.keep_original_prob

            elif str(key) == "icd10":
                self._rand_state["drop_icd10"] = self.R.random() < self.drop_prob

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        ret = dict()
        # deep copy all the unmodified data
        for key in set(data.keys()).difference(set(self.keys)):
            ret[key] = deepcopy(data[key])

        self.randomize(data)
        
        findings = ""
        impressions = ""
        icd10 = ""

        for key in self.keys:
            if str(key) == "findings":
                texts = data.get(key, [])
                if not texts or self._rand_state.get("drop_findings", False):
                    continue

                if len(texts) == 1 or self._rand_state.get("keep_findings_original", True):
                    text = texts[0]
                else:
                    text = self.R.choice(texts[1:])
                findings = f"Findings: {text}\n".replace("Impressions", "").replace("impressions", "")
            
            elif str(key) == "impressions":
                texts = data.get(key, [])
                if not texts:
                    continue

                if len(texts) == 1 or self._rand_state.get("keep_impressions_original", True):
                    text = texts[0]
                else:
                    text = self.R.choice(texts[1:])
                impressions = f"Impressions: {text}\n"
            
            elif str(key) == "icd10":
                codes = data.get(key, [])
                if isinstance(codes, str):
                    codes = codes.split(";")
                if not isinstance(codes, list) or not codes or self._rand_state.get("drop_icd10", False):
                    continue
                
                num_codes = len(codes) if self.max_num_icd10 < 0 else min(self.max_num_icd10, len(codes))
                if len(codes) <= self.max_num_icd10:
                    selected_codes = codes
                else:
                    selected_codes = self.R.choice(codes, size=num_codes, replace=False)
                icd10 = f"ICD10: {'; '.join(selected_codes)}\n"

        ret["report"] = f"{findings}{impressions}{icd10}"
        return ret


# class GenerateReportTransform(Randomizable, MapTransform):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         max_num_icd10=20,
#         likelihood_original=0.5,
#         drop_chance=0.3,
#         allow_missing_keys: bool = False,
#     ):
#         super().__init__(keys, allow_missing_keys)
#         self.max_num_icd10 = max_num_icd10
#         self.likelihood_original = likelihood_original
#         self.drop_chance = drop_chance

#         # Random states (purely indices/flags)
#         self.drop_findings = False
#         self.drop_icd10 = False
#         self.finding_idx = None
#         self.impression_idx = None
#         self.icd10_indices = []

#     def randomize(self, data):
#         findings = data.get("findings", [])
#         impressions = data.get("impressions", [])
#         icd10_codes = data.get("icd10", [])

#         if isinstance(icd10_codes, str):
#             icd10_codes = icd10_codes.split(";")
#         if not isinstance(icd10_codes, list):
#             icd10_codes = []

#         self.drop_findings = self.R.random() < self.drop_chance
#         self.drop_icd10 = self.R.random() < self.drop_chance
#         self.finding_idx = None
#         self.impression_idx = None
#         self.icd10_indices = []

#         if not self.drop_findings and findings:
#             num_elements = len(findings)
#             if num_elements == 1:
#                 self.finding_idx = 0
#             else:
#                 weights = [self.likelihood_original] + [(1 - self.likelihood_original) / (num_elements - 1)] * (num_elements - 1)
#                 self.finding_idx = int(self.R.choice(np.arange(num_elements), p=weights))

#         if impressions:
#             num_elements = len(impressions)
#             if num_elements == 1:
#                 self.impression_idx = 0
#             else:
#                 weights = [self.likelihood_original] + [(1 - self.likelihood_original) / (num_elements - 1)] * (num_elements - 1)
#                 self.impression_idx = int(self.R.choice(np.arange(num_elements), p=weights))

#         if not self.drop_icd10 and icd10_codes:
#             num_codes = min(self.max_num_icd10, len(icd10_codes))
#             self.icd10_indices = self.R.choice(len(icd10_codes), size=num_codes, replace=False).tolist()

#     def __call__(self, data):
#         self.randomize(data)

#         findings = data.get("findings", [])
#         impressions = data.get("impressions", [])
#         icd10_codes = data.get("icd10", [])

#         if isinstance(icd10_codes, str):
#             icd10_codes = icd10_codes.split(";")
#         if not isinstance(icd10_codes, list):
#             icd10_codes = []

#         report = ""

#         if self.finding_idx is not None and self.finding_idx < len(findings):
#             finding = findings[self.finding_idx].replace("Impressions", "").replace("impressions", "")
#             report += f"Findings: {finding}\n"

#         if self.impression_idx is not None and self.impression_idx < len(impressions):
#             impression = impressions[self.impression_idx]
#             report += f"Impressions: {impression}\n"

#         if self.icd10_indices:
#             selected_icd10 = [icd10_codes[i] for i in self.icd10_indices if i < len(icd10_codes)]
#             report += f"ICD10: {'; '.join(selected_icd10)}\n"

#         data["report"] = report
#         return data