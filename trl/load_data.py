import json
import os

from datasets import Dataset
from transformers import AutoTokenizer

#region Dummy dataset
# train_data = [
#     {
#         "prompt": [{"role": "user", "content": "What color is the sky?"}], 
#         "chosen": [{"role": "assistant", "content": "It is blue."}], 
#         "rejected": [{"role": "assistant", "content": "It is green."}]
#     },
#     {
#         "prompt": [{"role": "user", "content": "What is the capital of France?"}], 
#         "chosen": [{"role": "assistant", "content": "The capital of France is Paris."}], 
#         "rejected": [{"role": "assistant", "content": "The capital of France is London."}]
#     },
#     {
#         "prompt": [{"role": "user", "content": "What is the largest mammal?"}], 
#         "chosen": [{"role": "assistant", "content": "The blue whale is the largest mammal on Earth."}], 
#         "rejected": [{"role": "assistant", "content": "The elephant is the largest mammal."}]
#     }
# ]

# val_data = [
#     {
#         "prompt": [{"role": "user", "content": "What is the capital of Germany?"}], 
#         "chosen": [{"role": "assistant", "content": "The capital of Germany is Berlin."}], 
#         "rejected": [{"role": "assistant", "content": "The capital of Germany is Munich."}]
#     },
#     {
#         "prompt": [{"role": "user", "content": "What is the largest planet in our solar system?"}], 
#         "chosen": [{"role": "assistant", "content": "The largest planet in our solar system is Jupiter."}], 
#         "rejected": [{"role": "assistant", "content": "The largest planet in our solar system is Saturn."}]
#     }
# ]
#endregion

# load the data form file at given path and split into train and validation data
def load_train_val_data(file_path, split_ratio=0.9):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    # Split the data into train and validation sets
    train_data = data[:int(len(data) * split_ratio)]
    val_data = data[int(len(data) * split_ratio):]
    return train_data, val_data

# load data from the file "/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/bad_lang_examples_filtered.jsonl"
lang_train_data, lang_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/bad_lang_examples_filtered.jsonl")

# load data from the file "/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/short_examples_filtered.jsonl"
short_train_data, short_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/short_examples_filtered.jsonl")

# load data from the file "/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/choose_examples_filtered.jsonl"
choose_train_data, choose_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/choose_examples_filtered.jsonl")



# merge the three datasets into one
train_data = lang_train_data + short_train_data + choose_train_data
val_data = lang_val_data + short_val_data + choose_val_data

# remove the "src" field from each example
for example in train_data:
    del example["src"]
for example in val_data:
    del example["src"]

# replace the raw strings with { "role": "user/assistant", "content": "..." } objects
for example in train_data:
    example["prompt"] = [{"role": "user", "content": example["prompt"].replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model", "")}]
    example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
    example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]


# put the data into a dataset object
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
print("Created datasets")
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))
