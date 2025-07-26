import json
from transformers import AutoTokenizer
import pandas as pd

CURRICULUM_DATA = False

print("About to load tokenizer")

# 1) load your tokenizer
model_path = "cjvt/GaMS-9B-Instruct"                           # Path to the model
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, add_eos_token=True)    # Tokenizer for the model
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Loaded tokenizer")

def load_train_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

# 2) define a helper to turn the message-list into one string
def _flatten_messages(msg_list):
    return msg_list
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
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
        "src": example["src"],
        "prompt_length":      len(enc_prompt["input_ids"]),
        "chosen_full_length": len(enc_chosen["input_ids"]),
        "rejected_full_length": len(enc_reject["input_ids"]),
        "chosen_score": example["chosen_score"] if "chosen_score" in example else 0.5,
        "rejected_score": example["rejected_score"] if "rejected_score" in example else 0,
    }

def back_to_preference_format(example):
    if CURRICULUM_DATA:
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "src": example["src"],
            "chosen_score": example["chosen_score"],
            "rejected_score": example["rejected_score"],
        }
    else:
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "src": example["src"],
        }


path_prefix = "/workspace/preference_data"

files =[
    # "bad_lang_examples.jsonl",
    # "choose_examples.jsonl",
    # "short_examples.jsonl",
    # "bad_format_examples.jsonl",

    # "choose_examples_0.jsonl",

    "bad_lang_examples_1.jsonl",
    "choose_examples_1.jsonl",
    "short_examples_1.jsonl",
    "bad_format_examples_1.jsonl",

    "bad_lang_examples_2.jsonl",
    "choose_examples_2.jsonl",
    "short_examples_2.jsonl",
    "bad_format_examples_2.jsonl",
]
for file_path in files:
    train_data = load_train_data(f"{path_prefix}/raw_data/{file_path}")

    # 4) run it on your Dataset
    print(f"Processing {file_path} with {len(train_data)} examples")
    train_data = [ add_token_counts(x) for x in train_data]
    train_data = [ x for x in train_data if x["chosen_full_length"] < 3000 and x["rejected_full_length"] < 2040 ] # 2040 because of the few special tokens
    train_data = [ back_to_preference_format(x) for x in train_data]
    print(f"Filtered to {len(train_data)} examples")

    train_data = pd.DataFrame(train_data)

    # save the data to a file named f"{file_path without .jsonl}_filtered.jsonl"
    train_data.to_json(
        f'{path_prefix}/{"filtered_data" if not CURRICULUM_DATA else "curriculum_data"}/{file_path[:-6]}.jsonl', 
        orient="records", lines=True, force_ascii=False
    )
    print("-" * 40)
