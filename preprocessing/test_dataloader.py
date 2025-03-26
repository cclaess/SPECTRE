from spectre.data import CTRateDataset, MerlinDataset, VisionLanguageDataset
from spectre.ssl.transforms.clip_transform import GenerateReportTransform
import os

if __name__ == "__main__":

    merlin_data_path = r"C:/Users/20195435/Documents/TUe/SPECTRE"

    # Initialize the dataset
    dataset = MerlinDataset(
        data_dir=merlin_data_path,
        include_reports=True,
        transform=None
    )

    print(f"Number of samples: {len(dataset)}")

    print(dataset[0])

    # Initialize the dataset
    merlin_dataset = MerlinDataset(
        data_dir=merlin_data_path,
        include_reports=True,
        subset="train",	
        transform=None)

    ctrate_dataset = CTRateDataset(
        data_dir=merlin_data_path,
        include_reports=True,
        subset="train",
        transform=None)

    vision_language_dataset = VisionLanguageDataset(
        merlin_data=merlin_dataset,
        ctrate_data=ctrate_dataset,
        transform= None # GenerateReportTransform(keys=("findings", "impressions", "icd10"), icd10_range_lower=0.1, likelyhood_original=0.5, allow_missing_keys=True)

    )

   


    print(f"Number of samples: {len(vision_language_dataset)}")

    print(vision_language_dataset[0])

    vision_language_dataset = VisionLanguageDataset(
        merlin_data=merlin_dataset,
        ctrate_data=ctrate_dataset,
        transform=  GenerateReportTransform(keys=("findings", "impressions", "icd10"), icd10_range_lower=0.1, likelyhood_original=0.5, allow_missing_keys=True)

    )

    print(f"Number of samples: {len(vision_language_dataset)}")

    print(vision_language_dataset[0]["report"])

    


