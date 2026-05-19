from copy import deepcopy
from typing import Any, Mapping, Hashable

MONAI_IMPORT_ERROR = None
try:
    from monai.config import KeysCollection
    from monai.transforms import MapTransform, Randomizable
except ImportError as e:
    KeysCollection = Any  # type: ignore
    MONAI_IMPORT_ERROR = e


if MONAI_IMPORT_ERROR is None:
    _BaseClass = type("_BaseClass", (Randomizable, MapTransform), {})
else:
    _BaseClass = object


class RandomReportTransformd(_BaseClass):
    def __init__(
        self,
        keys: KeysCollection,
        max_num_icd10=20,
        keep_original_prob=0.5,
        drop_prob=0.3,
        allow_missing_keys: bool = False,
    ):
        if MONAI_IMPORT_ERROR is not None:
            raise ImportError(
                "MONAI is required to use RandomReportTransformd but not installed. "
                "Please install MONAI to use this transform."
            ) from MONAI_IMPORT_ERROR
        
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
