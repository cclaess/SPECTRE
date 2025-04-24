import pandas as pd
import os
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from spectre.data.datasets import MedicalDataset

data_path = r"C:\Users\20195435\Downloads\CT-RATE"

reports_path = os.path.join(data_path, "radiology_text_reports", "train_reports.csv")


dataset = MedicalDataset(reports_path)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# complete dataset to list
list_dataset = list(dataset)

print(dataset[0])
exit()

# Choose BPE as the base model (robust for subword tokenization)
tokenizer = Tokenizer(models.BPE())

# Set up a normalization pipeline to handle unicode, lowercasing, and stripping accents
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

# Use a whitespace pre-tokenizer to initially split on spaces (this can be extended for more complex tokenization)
tokenizer.pre_tokenizer = Whitespace()

# Set up the decoder so the tokenizer can convert token ids back into a string
tokenizer.decoder = decoders.BPEDecoder()

# Define a list of special tokens that are common for transformer models
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Create a trainer for BPE. Adjust the vocab_size based on your data and downstream needs.
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=special_tokens)

# Define an iterator over your dataset.
# Here, we assume that 'data_loader' is a prebuilt dataloader that yields batches of text strings.
def batch_iterator():
    for batch in data_loader:
        # 'batch' is assumed to be an iterable of strings
        for text in batch:
            yield text

# Train the tokenizer on your dataset using the iterator.
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# Save the trained tokenizer to a JSON file.
tokenizer.save("spectre_tokenizer.json")

# (Optional) If you plan to use this tokenizer with Hugging Face Transformers,
# you can wrap it with PreTrainedTokenizerFast.
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="spectre_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Save the fast tokenizer in the Hugging Face format.
fast_tokenizer.save_pretrained("spectre_tokenizer")