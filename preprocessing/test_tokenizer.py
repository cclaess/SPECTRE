from transformers import AutoTokenizer
import pandas as pd
import os
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from spectre.data.datasets import MedicalDataset



def evaluate_tokenizer_roundtrip(tokenizer, texts):
    """
    Evaluates how well the given Hugging Face tokenizer can encode and decode text.
    
    This function encodes each text (with special tokens included) by calling the tokenizer,
    extracts the input IDs, then decodes them back into text (skipping special tokens), and computes
    the percentage of texts where the roundtrip exactly recovers the original text.
    
    Args:
        tokenizer: A Hugging Face tokenizer (e.g., from AutoTokenizer).
        texts (list of str): List of texts to evaluate.
    
    Returns:
        float: The roundtrip accuracy as a fraction.
    """
    encoded_texts = []
    
    # Use the __call__ method to obtain the encoded outputs (dictionary with "input_ids").
    for text in texts:
        
        encoding = tokenizer(str(text), add_special_tokens=True)
        encoded_texts.append(encoding["input_ids"])
    
    # Decode the list of encoded token IDs.
    decoded_texts = tokenizer.batch_decode(encoded_texts, skip_special_tokens=True)
    
    correct = 0
    total = len(texts)

    incorrect_og = []
    incorrect_decoded = []
    
    for original, decoded in zip(texts, decoded_texts):
        # Compare after stripping potential whitespace differences.
        decoded = str(decoded)
        if original.strip().lower() == decoded.strip().lower():
            correct += 1
        else:

            incorrect_og.append(original)
            incorrect_decoded.append(decoded)
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"Roundtrip Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def count_num_tokens(text, tokenizer):
    """
    Count the number of tokens in the given text using the provided tokenizer.
    
    Args:
        text (str): The text to tokenize.
        tokenizer: A Hugging Face tokenizer (e.g., from AutoTokenizer).
    
    Returns:
        int: The number of tokens in the text.
    """
    encoding = tokenizer(str(text), add_special_tokens=True)
    return len(encoding["input_ids"])


def token_statistics(tokenizer, texts):
    """
    Compute statistics about the tokenization of the given texts using the provided tokenizer.
    
    Args:
        tokenizer: A Hugging Face tokenizer (e.g., from AutoTokenizer).
        texts (list of str): List of texts to analyze.
    
    Returns:
        dict: A dictionary with the following statistics:
            - "num_texts": The number of texts analyzed.
            - "num_tokens": The total number of tokens across all texts.
            - "avg_tokens": The average number of tokens per text.
            - "max_tokens": The maximum number of tokens in a single text.
    """
    num_texts = len(texts)
    num_tokens = 0
    max_tokens = 0
    
    for text in texts:
        num_tokens += count_num_tokens(text, tokenizer)
        max_tokens = max(max_tokens, count_num_tokens(text, tokenizer))
    
    avg_tokens = num_tokens / num_texts if num_texts > 0 else 0.0
    
    return {
        "num_texts": num_texts,
        "num_tokens": num_tokens,
        "avg_tokens": avg_tokens,
        "max_tokens": max_tokens
    }

def list_of_codes_to_long_desc(list_of_codes, icd10_dict):
    """
    Convert a list of ICD-10 codes to a list of long descriptions.
    
    Args:
        list_of_codes (list of str): List of ICD-10 codes.
        icd10_dict (dict): Dictionary mapping ICD-10 codes to long descriptions.
    
    Returns:
        list of str: List of long descriptions corresponding to the input codes.
    """
    new_list = []
    for case_codes in list_of_codes:
        long_desc = []

        cases_not_found = []
        list_of_case_codes = case_codes.split(",").strip()
        for code in list_of_case_codes:
            if code in icd10_dict:
                long_desc.append(icd10_dict[code])
            else:
                cases_not_found.append(code)

        # create a string with all the long descriptions
        long_desc_str = ", ".join(long_desc)
        new_list.append(long_desc_str)       
    return new_list

# Example usage:
if __name__ == "__main__":
    

    # Load a tokenizer. This could be any Hugging Face AutoTokenizer.
    # tokenizer_list = ["microsoft/BiomedVLP-CXR-BERT-specialized"] #, "microsoft/BiomedVLP-CXR-BERT-general", "Qwen/Qwen2.5-7B-Instruct"]
    tokenizer_list = ["Qwen/Qwen2.5-7B-Instruct"]
    # Define some sample texts to test.
    # sample_texts = [
    #     "This is a test sentence.",
    #     "Hugging Face tokenizers are awesome!",
    #     "Medical data often contains complex terminology.",
    #     "Ensure your tokenizer can handle various punctuation, e.g., commas, periods, and hyphens."
    # ]

    # data_path = r"C:\Users\20195435\Downloads\CT-RATE"

    # reports_path = os.path.join(data_path, "radiology_text_reports", "train_reports.csv")


    # dataset = MedicalDataset(reports_path)
    # sample_texts = list(dataset)
    # sample_texts = flatten_extend(sample_texts)
    # # data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # # convert all elements in list to string
    # sample_texts = [str(i) for i in sample_texts]
    # resutls = {}
        
    # # Evaluate the tokenizer roundtrip accuracy.
    # for tokenizer_name in tokenizer_list:
    #     print(f"Evaluating tokenizer: {tokenizer_name}")
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    #     resutls[tokenizer_name] = evaluate_tokenizer_roundtrip(tokenizer, sample_texts)
    # # evaluate_tokenizer_roundtrip(tokenizer, sample_texts)

    # print(resutls)

    merlin_data_prath = r"C:\Users\20195435\Downloads\reports_final_updated.xlsx"
    ICD10_path = r"C:\Users\20195435\Downloads\section111validicd10-jan2025_0.xlsx"
    icd10_data = pd.read_excel(ICD10_path, engine='openpyxl')

    # save ICD10 codes in a dictionary with CODE as key and LONG DESCRIPTION (VALID ICD-10 FY2025) as value
    icd10_dict = {}
    for index, row in icd10_data.iterrows():
        icd10_dict[row["CODE"]] = row["LONG DESCRIPTION (VALID ICD-10 FY2025)"]    

    merlin_data = pd.read_excel(merlin_data_prath, engine='openpyxl')

    merlin_data_findings = merlin_data["Findings"].tolist()
    merlin_data_icd10 = merlin_data["ICD10 Code"].tolist()

    # for each code, get the long description from the dictionary
    

    merlin_data_split = merlin_data["Split"].tolist()

    merlin_data_findings = [str(i) for i in merlin_data_findings]

    # get token statistics
    for tokenizer_name in tokenizer_list:
        print(f"Token statistics for tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        stats = token_statistics(tokenizer, merlin_data)
        print(stats)


