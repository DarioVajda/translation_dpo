import json
import os

from datasets import Dataset
# from transformers import AutoTokenizer

local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("ACCELERATE_LOCAL_RANK", 0)))
should_print = not local_rank or local_rank == 0

# load the data form file at given path and split into train and validation data
def load_train_val_data(file_path, split_ratio=0.9):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    # Split the data into train and validation sets
    train_data = data[:int(len(data) * split_ratio)]
    val_data = data[int(len(data) * split_ratio):]
    return train_data, val_data

# Loading the evaluation data from the old version of the dataset (for fair comparison)
_, lang_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_lang_examples.jsonl")
_, short_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples.jsonl")
_, choose_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/choose_examples.jsonl")
_, format_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples.jsonl")


# Loading the training data from the new version of the dataset
lang_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_lang_examples.jsonl")
lang_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_lang_examples_1.jsonl", split_ratio=1)
lang_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_lang_examples_2.jsonl", split_ratio=1)

short_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples.jsonl")
short_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples_1.jsonl", split_ratio=1)
short_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples_2.jsonl", split_ratio=1)

choose_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/choose_examples_0.jsonl") # Using _0 file because it has bigger requirement for the comet score difference then the old version
choose_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/choose_examples_1.jsonl", split_ratio=1)
choose_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/choose_examples_2.jsonl", split_ratio=1)

format_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples.jsonl")
format_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples_1.jsonl", split_ratio=1)
format_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples_2.jsonl", split_ratio=1)

lang_train_data = lang_train_data0 + lang_train_data1 + lang_train_data2
short_train_data = short_train_data0 + short_train_data1 + short_train_data2
choose_train_data = choose_train_data0 + choose_train_data1 + choose_train_data2
format_train_data = format_train_data0 + format_train_data1 + format_train_data2


if should_print: print("[load_data.py]: Training data of type 'bad_lang_examples':   ", len(lang_train_data))
if should_print: print("[load_data.py]: Training data of type 'short_examples':      ", len(short_train_data))
if should_print: print("[load_data.py]: Training data of type 'choose_examples':     ", len(choose_train_data))
if should_print: print("[load_data.py]: Training data of type 'bad_format_examples': ", len(format_train_data))

# merge the three datasets into one
train_data = lang_train_data + short_train_data + choose_train_data + format_train_data
val_data = lang_val_data + short_val_data + choose_val_data # + format_val_data

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
for example in val_data:
    example["prompt"] = [{"role": "user", "content": example["prompt"].replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model", "")}]
    example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
    example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]


# print the first 3 examples of the training data
# if should_print: print("First 3 training examples:")
# for i in range(3):
#     if should_print: print(f"Example {i+1}:")
#     if should_print: print("Prompt:", train_data[i]["prompt"])
#     if should_print: print("Chosen:", train_data[i]["chosen"])
#     if should_print: print("Rejected:", train_data[i]["rejected"])
#     if should_print: print()


# put the data into a dataset object
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
# if should_print: print("Created datasets")
if should_print: print("[load_data.py]: Number of training examples:", len(train_dataset))
if should_print: print("[load_data.py]: Number of validation examples:", len(val_dataset))


# # print the percentage of each type of training data
# print("Percentage of bad_lang_examples in training data: ", len(lang_train_data) / len(train_dataset) * 100)
# print("Percentage of short_examples in training data: ", len(short_train_data) / len(train_dataset) * 100)
# print("Percentage of choose_examples in training data: ", len(choose_train_data) / len(train_dataset) * 100)
# print("Percentage of bad_format_examples in training data: ", len(format_train_data) / len(train_dataset) * 100)
