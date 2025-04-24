import pandas as pd
import os


ICD10_path = r"C:\Users\20195435\Downloads\ICD_10jan2025_0.xlsx"
icd10_data = pd.read_excel(ICD10_path, engine='openpyxl')

# create dict with CODE as key and LONG DESCRIPTION (VALID ICD-10 FY2025) as value
icd10_dict = {}
for index, row in icd10_data.iterrows():
    icd10_dict[row["CODE"]] = row["LONG DESCRIPTION (VALID ICD-10 FY2025)"]

ICD10_excluded_path = r"C:\Users\20195435\Downloads\section111excludedicd10-jan2025_0.xlsx"
icd10_excluded_data = pd.read_excel(ICD10_excluded_path, engine='openpyxl')

# create dict with CODE as key and the second column as value
icd10_excluded_dict = {}
for index, row in icd10_excluded_data.iterrows():
    icd10_excluded_dict[row["CODE"]] = row.iloc[1]

# step through excluded data and add it to the icd10_dict if not already present
for code, desc in icd10_excluded_dict.items():
    if code not in icd10_dict:
        icd10_dict[code] = desc

# save the dictionary to a excel file
df = pd.DataFrame(icd10_dict.items(), columns=["CODE", "LONG DESCRIPTION"])
df.to_excel("ICD10_dict.xlsx", index=False)
print("ICD10_dict.xlsx saved")

