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
    start_time = time.time()
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

        response = response.json()
        total_time = time.time() - start_time
        print(total_time)
        response["response_time"] = total_time

        return response
    except requests.RequestException as e:
        logging.error("Error during API request: %s", e)
        return None

def general_augment(text, finding=True):
    if finding:
        prompt = (
            "Below is a CT scan report presenting patient findings. "
            "Generate an alternative version that summarizes and shortens the text while preserving every factual detail. "
            "Use your own knowledge to present an accurate, alternative representation without altering any key information. Avoid adding markdown or any other formatting. "
            "Here is the report:\n\n" +
            text
        )
    else:
        prompt = (
            "Below is a list of impressions derived from a patient CT scan and report. "
            "Articulate an accurate, alternative representation without altering any key information. Avoid adding markdown and keep the original numbering. "
            "Here is the list:\n\n" +
            text
        )
    response = augment_text(prompt)
    if response is None:
        return None
    content = response["choices"][0]["message"]["content"]
    return content

def list_of_codes_to_long_desc(list_of_codes, icd10_dict):
    """
    Convert a list of ICD-10 codes to a list of long descriptions.
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
    """
    parts = findings.split("\n")
    if len(parts) == 1:
        return clean_finding(parts[0]), None
    else:
        return clean_finding(parts[0]), clean_impression(parts[1:])

def sanity_check_impressions(ct_data_impressions):
    """
    Check that each line of the impressions starts with a number.
    """
    for impression in ct_data_impressions:
        for line in impression.split("\n"):
            if line and not line[0].isdigit():
                logging.error("Impression does not start with a number: %s", line)
                raise ValueError("Impression does not start with a number")
    return True

def parse_codes(data):
    """
    Parse ICD-10 codes and descriptions from a text file.
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
    """
    return finding.replace("FINDINGS:", "").strip()

def clean_impression(impression_lines):
    """
    Clean and join impression lines.
    """
    return "\n".join(impression_lines).strip()

def save_intermediate(df, filename, intermediate_dir):
    """
    Save the intermediate DataFrame to an Excel file.
    """
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    filepath = os.path.join(intermediate_dir, filename)
    df.to_excel(filepath, index=False)
    logging.info("Intermediate file saved: %s", filepath)

def has_valid_content(value):
    """
    Check whether a cell contains valid string content.
    Returns True if the value is not null and contains non-whitespace characters.
    """
    if pd.isnull(value):
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if len(str(value)) == 0:
        return False
    return True

def apply_with_checkpoint(df, source_column, target_column, func, checkpoint_interval, checkpoint_filename, extra_kwargs=None):
    """
    Apply a function row by row on a DataFrame column, starting from the first row where the target column is either null or has no string content.
    Saves a checkpoint of the DataFrame after every checkpoint_interval processed rows.
    
    Args:
        df (pd.DataFrame): The DataFrame being processed.
        source_column (str): The column containing the source text.
        target_column (str): The column where augmented text is stored.
        func (callable): The augmentation function to apply.
        checkpoint_interval (int): Number of rows to process before saving a checkpoint.
        checkpoint_filename (str): Filename for the checkpoint.
        extra_kwargs (dict, optional): Extra keyword arguments for the function.
    
    Returns:
        pd.Series: The updated target column.
    """
    # Determine the first index in target_column that is either null or empty after stripping
    missing_indices = df.index[~df[target_column].apply(has_valid_content)]
    if missing_indices.empty:
        logging.info("All entries in column '%s' contain valid content. Skipping processing.", target_column)
        return df[target_column]
    print("missing_indices", missing_indices)
    start_idx = missing_indices[0]
    logging.info("Starting processing for column '%s' from row index %s", target_column, start_idx)
    processed_count = 0
    
    for idx in range(start_idx, len(df)):
        if has_valid_content(df.at[idx, target_column]):
            continue
        
        value = df.at[idx, source_column]
        if pd.isnull(value):
            augmented = value
        else:
            if extra_kwargs:
                augmented = func(value, **extra_kwargs)
            else:
                augmented = func(value)
        df.at[idx, target_column] = augmented
        processed_count += 1
        
        # Save checkpoint after every checkpoint_interval processed rows
        if processed_count % checkpoint_interval == 0:
            df.to_excel(checkpoint_filename, index=False)
            logging.info("Checkpoint saved for column '%s' after processing %d rows", target_column, processed_count)
    
    return df[target_column]

def main(args):
    setup_logging()
    
    # Read input data from Excel
    logging.info("Reading CT rate data from %s", args.ct_data_path)
    ct_data = pd.read_excel(args.ct_data_path, engine='openpyxl')
    
    # Optionally process only the first N rows (useful for testing)
    if args.head:
        ct_data = ct_data.head(args.head)
    
    # Save raw input data as an intermediate step
    save_intermediate(ct_data, "ct_data_raw.xlsx", args.intermediate_dir)
    
    # Augment findings with checkpointing and resume support
    logging.info("Starting augmentation for findings")
    for i in range(1, 2):
        col_name = f"Findings_{i}"
        checkpoint_file = os.path.join(args.intermediate_dir, f"checkpoint_{col_name}.xlsx")
        
        # If a checkpoint exists, load it and update the DataFrame
        if os.path.exists(checkpoint_file):
            logging.info("Checkpoint file '%s' found. Resuming augmentation for '%s'.", checkpoint_file, col_name)
            checkpoint_df = pd.read_excel(checkpoint_file, engine='openpyxl')
            if col_name in checkpoint_df.columns:
                ct_data.loc[checkpoint_df[col_name].apply(has_valid_content), col_name] = checkpoint_df.loc[checkpoint_df[col_name].apply(has_valid_content), col_name]
        
        ct_data[col_name] = apply_with_checkpoint(
            ct_data, 
            source_column="Findings_EN", 
            target_column=col_name, 
            func=general_augment, 
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_filename=checkpoint_file
        )
        logging.info("Augmentation for %s done", col_name)
        save_intermediate(ct_data, f"ct_data_{col_name}.xlsx", args.intermediate_dir)
    
    # Augment impressions with checkpointing and resume support; pass findings=False to the augmentation function
    logging.info("Starting augmentation for impressions")
    for i in range(1, 2):
        col_name = f"Impressions_{i}"
        checkpoint_file = os.path.join(args.intermediate_dir, f"checkpoint_{col_name}.xlsx")
        
        if os.path.exists(checkpoint_file):
            logging.info("Checkpoint file '%s' found. Resuming augmentation for '%s'.", checkpoint_file, col_name)
            checkpoint_df = pd.read_excel(checkpoint_file, engine='openpyxl')
            if col_name in checkpoint_df.columns:
                ct_data.loc[checkpoint_df[col_name].apply(has_valid_content), col_name] = checkpoint_df.loc[checkpoint_df[col_name].apply(has_valid_content), col_name]
        
        ct_data[col_name] = apply_with_checkpoint(
            ct_data,
            source_column="Impressions_EN",
            target_column=col_name,
            func=general_augment,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_filename=checkpoint_file,
            extra_kwargs={'findings': False}
        )
        logging.info("Augmentation for %s done", col_name)
        save_intermediate(ct_data, f"ct_data_{col_name}.xlsx", args.intermediate_dir)
    
    # Save final DataFrame to output file
    ct_data.to_excel(args.output_path, index=False)
    logging.info("Final file saved: %s", args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Merlin data with ICD10 codes and text augmentation with checkpoints.")
    parser.add_argument("--ct_data_path", type=str, default=r"C:\Users\20195435\Documents\TUe\SPECTRE\CT-RATE\train_reports_updated.xlsx",
                        help="Path to the Merlin Excel data file.")
    parser.add_argument("--output_path", type=str, default=r"C:\Users\20195435\Documents\TUe\SPECTRE\CT-RATE\train_reports_updated_gen.xlsx",
                        help="Path for the final output Excel file.")
    parser.add_argument("--intermediate_dir", type=str, default=r"C:\Users\20195435\Documents\TUe\SPECTRE\CT-RATE\intermediate",
                        help="Directory to save intermediate Excel files.")
    parser.add_argument("--head", type=int, default=None,
                        help="Process only the first N rows (for testing).")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Number of rows to process before saving a checkpoint.")
    args = parser.parse_args()
    
    main(args)
