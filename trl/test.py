import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def tree(dir_path, prefix=""):
    entries = sorted(os.listdir(dir_path))
    for i, name in enumerate(entries):
        path = os.path.join(dir_path, name)
        connector = "└── " if i == len(entries)-1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = "    " if i == len(entries)-1 else "│   "
            tree(path, prefix + extension)

# tree("/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/training_run/r-128_lr-5e-06_b-0.25/checkpoint-90")


# Load jsonl file from "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_gams9b_dpo_translation.jsonl"
import json
import os

def load_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

file_path = "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_gams9b_dpo_translation.jsonl"
# file_path = "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_eurollm9b_translation.jsonl"
data = load_jsonl(file_path)

# Print the first entry
index = 5
print(data[index].keys())
print(data[index]['Prompt'])
print('-'*30)
print(data[index]['sl_translation'])
