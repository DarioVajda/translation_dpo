import json
import os

from datasets import Dataset
# from transformers import AutoTokenizer

local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("ACCELERATE_LOCAL_RANK", 0)))
should_print = not local_rank or local_rank == 0

# load the data form file at given path and split into train and validation data
def load_train_val_data(file_path, split_ratio=0.9):
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        # Split the data into train and validation sets
        train_data = data[:int(len(data) * split_ratio)]
        val_data = data[int(len(data) * split_ratio):]
        return train_data, val_data
    except:
        if should_print: print(f"[load_data.py]: Error loading {file_path}")
        return [], []

# Loading the evaluation data from the old version of the dataset (for fair comparison)
_, lang_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_lang_examples.jsonl")
_, short_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples.jsonl")
_, choose_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/choose_examples.jsonl")
_, format_val_data = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples.jsonl")

#region old data
# Loading the training data from the new version of the dataset
# lang_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_lang_examples.jsonl")
# lang_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_lang_examples_1.jsonl", split_ratio=1)
# lang_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_lang_examples_2.jsonl", split_ratio=1)
# lang_train_data_ccnews1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data_ccnews/bad_lang_examples_1.jsonl", split_ratio=1)
# lang_train_data_ccnews2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data_ccnews/bad_lang_examples_2.jsonl", split_ratio=1)

# short_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples.jsonl")
# short_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples_1.jsonl", split_ratio=1)
# short_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples_2.jsonl", split_ratio=1)
# short_train_data_ccnews1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data_ccnews/short_examples_1.jsonl", split_ratio=1)
# short_train_data_ccnews2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data_ccnews/short_examples_2.jsonl", split_ratio=1)

# choose_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/choose_examples_0.jsonl") # Using _0 file because it has bigger requirement for the comet score difference then the old version
# choose_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/choose_examples_1.jsonl", split_ratio=1)
# choose_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/choose_examples_2.jsonl", split_ratio=1)
# choose_train_data_ccnews1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data_ccnews/choose_examples_1.jsonl", split_ratio=1)
# choose_train_data_ccnews2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data_ccnews/choose_examples_2.jsonl", split_ratio=1)

# format_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples.jsonl")
# format_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples_1.jsonl", split_ratio=1)
# format_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples_2.jsonl", split_ratio=1)
# format_train_data_ccnews1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data_ccnews/bad_format_examples_1.jsonl", split_ratio=1)
# format_train_data_ccnews2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data_ccnews/bad_format_examples_2.jsonl", split_ratio=1)

# lang_train_data = lang_train_data0 + lang_train_data1 + lang_train_data2 + lang_train_data_ccnews1 + lang_train_data_ccnews2
# short_train_data = short_train_data0 + short_train_data1 + short_train_data2 + short_train_data_ccnews1 + short_train_data_ccnews2
# choose_train_data = choose_train_data0 + choose_train_data1 + choose_train_data2 + choose_train_data_ccnews1 + choose_train_data_ccnews2
# format_train_data = format_train_data0 + format_train_data1 + format_train_data2 + format_train_data_ccnews1 + format_train_data_ccnews2
#endregion

def load_train_data(dir_path, data_id):
    try:
        lang_train_data, _ = load_train_val_data(os.path.join(dir_path, f"bad_lang_examples_id{data_id}.jsonl"), split_ratio=1)
        short_train_data, _ = load_train_val_data(os.path.join(dir_path, f"short_examples_id{data_id}.jsonl"), split_ratio=1)
        choose_train_data, _ = load_train_val_data(os.path.join(dir_path, f"choose_examples_id{data_id}.jsonl"), split_ratio=1)
        format_train_data, _ = load_train_val_data(os.path.join(dir_path, f"bad_format_examples_id{data_id}.jsonl"), split_ratio=1)
        return lang_train_data, short_train_data, choose_train_data, format_train_data
    except FileNotFoundError as e:
        if should_print: print(f"[load_data.py]: Warning: {e}. Skipping this dataset.")
        return [], [], [], []

def load_train_file(file_path):
    try:
        train_data, _ = load_train_val_data(file_path, split_ratio=1)
        return train_data
    except FileNotFoundError as e:
        if should_print: print(f"[load_data.py]: Warning: {e}. Skipping this file.")
        return []

dir_list = [
    # CC-News (0,1,2,3,4)
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/cc_news_0/preference_data', 0),
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/cc_news_1/preference_data', 1),
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/cc_news_2/preference_data', 2),
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/cc_news_3/preference_data', 3),
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/cc_news_4/preference_data', 4),

    # BookCorpus (0,1)
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/bookcorpus_0/preference_data', 0),
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/bookcorpus_1/preference_data', 1),

    # Wikipedia (0,2,3,4; wikipedia 1 is corrupted)
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/wikipedia_0/preference_data', 0),
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/wikipedia_2/preference_data', 2),
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/wikipedia_3/preference_data', 3),
    ('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/wikipedia_4/preference_data', 4),
]
dir_list = [] # ONLY USING THE UNIFIED DATASET NOW

file_list = [
    # '/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/large_training_dataset/markdown_from_frida/data_train.jsonl',
    "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/all_train_data.jsonl",
]

lang_train_data = []
short_train_data = []
choose_train_data = []
format_train_data = []
for dir_path, data_id in dir_list:
    curr_ltd, curr_std, curr_ctd, curr_ftd = load_train_data(dir_path, data_id)
    lang_train_data += curr_ltd
    short_train_data += curr_std
    choose_train_data += curr_ctd
    format_train_data += curr_ftd
    if should_print: print(f"[load_data.py]: Loaded data from {dir_path} with {len(curr_ltd) + len(curr_std) + len(curr_ctd) + len(curr_ftd)} examples.")

if should_print: 
    print("[load_data.py]: Training data of type 'bad_lang_examples':   ", len(lang_train_data))
    print("[load_data.py]: Training data of type 'short_examples':      ", len(short_train_data))
    print("[load_data.py]: Training data of type 'choose_examples':     ", len(choose_train_data))
    print("[load_data.py]: Training data of type 'bad_format_examples': ", len(format_train_data))

other_train_data = []
for file_path in file_list:
    curr_otd = load_train_file(file_path)
    other_train_data += curr_otd
if should_print: print("[load_data.py]: Training data of type 'other_examples':      ", len(other_train_data))

# merge the three datasets into one
train_data = lang_train_data + short_train_data + choose_train_data + format_train_data + other_train_data
val_data = lang_val_data + short_val_data + choose_val_data # + format_val_data

# remove the "src" field from each example
for example in train_data:
    if "src" in example: del example["src"]
for example in val_data:
    if "src" in example: del example["src"]

# !!! Not performing it here because it has already been done on Vega HPC !!!
# for example in train_data:
#     example["prompt"] = [{"role": "user", "content": example["prompt"].replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model", "")}]
#     example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
#     example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]
# replace the raw strings with { "role": "user/assistant", "content": "..." } objects
for example in val_data:
    example["prompt"] = [{"role": "user", "content": example["prompt"].replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model", "")}]
    example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
    example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]

if should_print: print(f"[load_data.py]: Total training data size: {len(train_data)}")
if should_print: print(f"[load_data.py]: Total validation data size: {len(val_data)}")