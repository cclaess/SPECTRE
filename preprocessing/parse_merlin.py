import pandas as pd
import os
import logging
import argparse
import numpy as np
import requests
import json
import time

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    

def augment_text(prompt):
    to = time.time()
    # Build the payload for the API call
    payload = {
        "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "messages": [
            {"role": "system", "content": "You are a helpful medical doctor."},
            {"role": "user", "content": prompt}
        ],
        "stop_token_ids": [128009, 128001]
    }
    
    # The endpoint provided in your description
    url = "http://10.88.23.8:8000/v1/chat/completions"
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad responses

        # add the response time to the response
        response = response.json()
        total_time = time.time() - to
        print(total_time)
        response["response_time"] = total_time

        return response
    except requests.RequestException as e:
        print("Error during API request:", e)
        return None
    
def general_augment(text, finding=True):
    if finding:
        prompt = (
            "Below is a CT scan report presenting patient findings. "
            "Generate an alternative version that summarizes and shortens the text while preserving every factual detail. "
            "Use your own knowledge to present an accurate, alternative representation without altering any key information. Avoid adding markdown or any other formatting."
            "Here is the report:\n\n"
            f"{text}"
        )
    else:
        prompt = (
            "Below is a list of impressions derived from a patient CT scan and report. "
            "Articulate an accurate, alternative representation without altering any key information. Avoid adding markdown and keep the original numbering."
            "Here is the list:\n\n"
            f"{text}"
        )
    response = augment_text(prompt)
    content = response["choices"][0]["message"]["content"]
    return content

def list_of_codes_to_long_desc(list_of_codes, icd10_dict):
    """
    Convert a list of ICD-10 codes to a list of long descriptions.
    
    Args:
        list_of_codes (list of str): List of ICD-10 codes.
        icd10_dict (dict): Dictionary mapping ICD-10 codes to long descriptions.
    
    Returns:
        tuple: (list of long descriptions, list of codes not found)
    """
    new_list = []
    cases_not_found = []
    for case_codes in list_of_codes:
        long_desc = []
        # Split multiple codes if separated by commas
        list_of_case_codes = str(case_codes).split(",")
        list_of_case_codes = [code.strip() for code in list_of_case_codes]
        for code in list_of_case_codes:
            # Standardize the code by removing dots
            code_clean = code.replace(".", "")
            if code_clean in icd10_dict:
                long_desc_text = icd10_dict[code_clean].strip()
                long_desc.append(long_desc_text)
            else:
                cases_not_found.append(code)
        long_desc_str = ", ".join(long_desc)
        new_list.append(long_desc_str)
    return new_list, cases_not_found

def split_findings(findings):
    """
    Split text into primary finding and impressions by newline.
    
    Args:
        findings (str): Text containing findings and impressions.
    
    Returns:
        tuple: (cleaned finding, cleaned impression or None)
    """
    parts = findings.split("\n")
    if len(parts) == 1:
        return clean_finding(parts[0]), None
    else:
        return clean_finding(parts[0]), clean_impression(parts[1:])

def sanity_check_impressions(merlin_data_impressions):
    """
    Check that each line of the impressions starts with a number.
    
    Args:
        merlin_data_impressions (iterable): Collection of impression strings.
    
    Returns:
        bool: True if all impressions pass the check.
    
    Raises:
        ValueError: If any line does not start with a number.
    """
    for impression in merlin_data_impressions:
        for line in impression.split("\n"):
            if line and not line[0].isdigit():
                logging.error("Impression does not start with a number: %s", line)
                raise ValueError("Impression does not start with a number")
    return True

def parse_codes(data):
    """
    Parse ICD-10 codes and descriptions from a text file.
    
    Args:
        data (str): Text data containing ICD-10 codes and their descriptions.
    
    Returns:
        dict: Dictionary mapping codes to descriptions.
    """
    code_dict = {}
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue  # skip empty lines
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            code, description = parts
            code_dict[code] = description
    return code_dict

def clean_finding(finding):
    """
    Clean the finding text by removing the 'FINDINGS:' label and trimming whitespace.
    
    Args:
        finding (str): The raw finding text.
    
    Returns:
        str: The cleaned finding.
    """
    return finding.replace("FINDINGS:", "").strip()

def clean_impression(impression_lines):
    """
    Clean and join impression lines.
    
    Args:
        impression_lines (list of str): Lines constituting the impression.
    
    Returns:
        str: The cleaned impression.
    """
    return "\n".join(impression_lines).strip()

def save_intermediate(df, filename, intermediate_dir):
    """
    Save the intermediate DataFrame to an Excel file.
    
    Args:
        df (pd.DataFrame): DataFrame to be saved.
        filename (str): Name of the file to save.
        intermediate_dir (str): Directory where the file will be stored.
    """
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    filepath = os.path.join(intermediate_dir, filename)
    df.to_excel(filepath, index=False)
    logging.info("Intermediate file saved: %s", filepath)

def apply_with_checkpoint(df, source_column, target_column, func, checkpoint_interval, checkpoint_filename, extra_kwargs=None):
    """
    Apply a function row by row on a DataFrame column, updating a target column,
    and save a checkpoint of the DataFrame after every checkpoint_interval rows.

    Args:
        df (pd.DataFrame): The DataFrame being processed.
        source_column (str): The name of the column to process.
        target_column (str): The name of the column to store results.
        func (callable): The function to apply (e.g., general_augment).
        checkpoint_interval (int): How many rows to process before saving a checkpoint.
        checkpoint_filename (str): File name for the checkpoint file.
        extra_kwargs (dict, optional): Extra keyword arguments for the function.

    Returns:
        pd.Series: The resulting augmented column.
    """
    results = []
    for idx, value in df[source_column].items():
        # Skip processing if the value is missing
        if pd.isnull(value):
            augmented = value
        else:
            if extra_kwargs:
                augmented = func(value, **extra_kwargs)
            else:
                augmented = func(value)
        results.append(augmented)
        df.at[idx, target_column] = augmented

        # Save a checkpoint after every checkpoint_interval rows
        if (idx + 1) % checkpoint_interval == 0:
            df.to_excel(checkpoint_filename, index=False)
            logging.info("Checkpoint saved for column '%s' after processing %d rows", target_column, idx+1)
    return pd.Series(results, index=df.index)

def main(args):
    setup_logging()
    
    # Read ICD-10 codes dictionary
    logging.info("Reading ICD10 codes from %s", args.icd10_path)
    with open(args.icd10_path, "r") as f:
        data = f.read()
    icd10_dict = parse_codes(data)
    
    # Read Merlin data from Excel
    logging.info("Reading Merlin data from %s", args.merlin_data_path)
    merlin_data = pd.read_excel(args.merlin_data_path, engine='openpyxl')
    
    # Optionally process only the first N rows (useful for testing)
    if args.head:
        merlin_data = merlin_data.head(args.head)
    
    # Save raw input data as an intermediate step
    save_intermediate(merlin_data, "merlin_data_raw.xlsx", args.intermediate_dir)
    
    # Convert ICD10 codes to long descriptions
    logging.info("Converting ICD10 codes to long descriptions")
    icd10_codes = merlin_data["ICD10 Code"].tolist()
    long_icd10_desc, cases_not_found = list_of_codes_to_long_desc(icd10_codes, icd10_dict)
    merlin_data["FULL_ICD10 Description"] = long_icd10_desc

    print(len(np.unique(cases_not_found)))
    
    save_intermediate(merlin_data, "merlin_data_with_icd10.xlsx", args.intermediate_dir)
    
    # Split the 'Findings' column into separate findings and impressions
    logging.info("Splitting findings and impressions")
    findings_list = merlin_data["Findings"].tolist()
    findings_separate = []
    impressions_separate = []
    for finding in findings_list:
        finding_text, impression_text = split_findings(finding)
        findings_separate.append(finding_text)
        impressions_separate.append(impression_text)
    
    merlin_data["Findings_0"] = findings_separate
    merlin_data["Impressions_0"] = impressions_separate
    save_intermediate(merlin_data, "merlin_data_split.xlsx", args.intermediate_dir)
    
    # Augment findings with checkpointing
    logging.info("Starting augmentation for findings")
    for i in range(1, 2):
        col_name = f"Findings_{i}"
        checkpoint_file = os.path.join(args.intermediate_dir, f"checkpoint_{col_name}.xlsx")
        merlin_data[col_name] = apply_with_checkpoint(
            merlin_data, 
            source_column="Findings_0", 
            target_column=col_name, 
            func=general_augment, 
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_filename=checkpoint_file
        )
        logging.info("Augmentation for %s done", col_name)
        save_intermediate(merlin_data, f"merlin_data_{col_name}.xlsx", args.intermediate_dir)
    
    # Augment impressions with checkpointing; pass findings=False to the augmentation function
    logging.info("Starting augmentation for impressions")
    for i in range(1, 2):
        col_name = f"Impressions_{i}"
        checkpoint_file = os.path.join(args.intermediate_dir, f"checkpoint_{col_name}.xlsx")
        merlin_data[col_name] = apply_with_checkpoint(
            merlin_data,
            source_column="Impressions_0",
            target_column=col_name,
            func=general_augment,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_filename=checkpoint_file,
            extra_kwargs={'findings': False}
        )
        logging.info("Augmentation for %s done", col_name)
        save_intermediate(merlin_data, f"merlin_data_{col_name}.xlsx", args.intermediate_dir)
    
    # Save final DataFrame to output file
    merlin_data.to_excel(args.output_path, index=False)
    logging.info("Final file saved: %s", args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Merlin data with ICD10 codes and text augmentation with checkpoints.")
    parser.add_argument("--merlin_data_path", type=str, default=r"C:\Users\20195435\Downloads\reports_final_updated.xlsx",
                        help="Path to the Merlin Excel data file.")
    parser.add_argument("--icd10_path", type=str, default=r"C:\Users\20195435\Downloads\Code-desciptions-April-2025\icd10cm-codes-April-2025.txt",
                        help="Path to the ICD10 codes text file.")
    parser.add_argument("--output_path", type=str, default="merlin_data_updated.xlsx",
                        help="Path for the final output Excel file.")
    parser.add_argument("--intermediate_dir", type=str, default="intermediate_steps",
                        help="Directory to save intermediate Excel files.")
    parser.add_argument("--head", type=int, default=None,
                        help="Process only the first N rows (for testing).")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Number of rows to process before saving a checkpoint.")
    args = parser.parse_args()
    
    main(args)
