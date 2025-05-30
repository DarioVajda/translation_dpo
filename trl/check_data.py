print("About to import libraries")
from transformers import AutoTokenizer
print("Imported libraries")
from datasets import Dataset
print("Imported datasets")
from load_data import train_dataset, val_dataset
print("Loaded data")

print("About to load tokenizer")

# 1) load your tokenizer
model_path = "cjvt/GaMS-9B-Instruct"                           # Path to the model
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, add_eos_token=True)    # Tokenizer for the model
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Loaded tokenizer")

# 2) define a helper to turn the message-list into one string
def _flatten_messages(msg_list):
    # msg_list is [{"role": "...", "content": "..."}]
    return " ".join([m["content"] for m in msg_list])

# 3) define a map function that returns the various token‐counts
def add_token_counts(example):
    prompt_text   = _flatten_messages(example["prompt"])
    chosen_text   = _flatten_messages(example["chosen"])
    rejected_text = _flatten_messages(example["rejected"])
    
    # tokenize without any truncation/padding
    enc_prompt = tokenizer(prompt_text,   truncation=False)
    enc_chosen = tokenizer(prompt_text + chosen_text,   truncation=False)
    enc_reject = tokenizer(prompt_text + rejected_text, truncation=False)
    
    return {
      "prompt_length":      len(enc_prompt["input_ids"]),
      "chosen_full_length": len(enc_chosen["input_ids"]),
      "rejected_full_length": len(enc_reject["input_ids"]),
    }

# 4) run it on your Dataset
train_dataset = train_dataset.map(add_token_counts)

# 5) now each example has three new columns:
#    prompt_length, chosen_full_length, rejected_full_length
#    you can inspect:
print(train_dataset[0])
#    or get summary stats:
import numpy as np
print("prompt  →",      np.mean(train_dataset["prompt_length"]),      np.max(train_dataset["prompt_length"]))
print("prompt+chosen →",np.mean(train_dataset["chosen_full_length"]),np.max(train_dataset["chosen_full_length"]))
print("prompt+reject →",np.mean(train_dataset["rejected_full_length"]),np.max(train_dataset["rejected_full_length"]))

# print out all percentile values for the token counts: 5%, 10%, 25%, 50%, 75%, 90%, 95%
print("prompt  →",      np.percentile(train_dataset["prompt_length"],      [5, 10, 25, 50, 75, 90, 95]))
print("prompt+chosen →",np.percentile(train_dataset["chosen_full_length"],[5, 10, 25, 50, 75, 90, 95]))
print("prompt+reject →",np.percentile(train_dataset["rejected_full_length"],[5, 10, 25, 50, 75, 90, 95]))

# What percentage of the data has a token count less than 2048 for both the chosen and rejected texts at once
print("Percentage of data with chosen+reject < 2048:", len([x for x in train_dataset if x["chosen_full_length"] < 2048 and x["rejected_full_length"] < 2048]) / len(train_dataset) * 100)

# plot the distribution of the token counts. Save the plot in a file
import matplotlib.pyplot as plt
import seaborn as sns

def plot_token_distribution(dataset, column_name, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset[column_name], bins=50, kde=True)
    plt.title(title)
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(f"/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/{column_name}.png")
    plt.close()
plot_token_distribution(train_dataset, "prompt_length", "Prompt Length Distribution")
plot_token_distribution(train_dataset, "chosen_full_length", "Chosen Length Distribution")
plot_token_distribution(train_dataset, "rejected_full_length", "Rejected Length Distribution")
