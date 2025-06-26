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
lang_train_data2 = [] # There was too much data of this type, so we will not use it in the training

short_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples.jsonl")
short_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples_1.jsonl", split_ratio=1)
short_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/short_examples_2.jsonl", split_ratio=1)

format_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples.jsonl")
format_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples_1.jsonl", split_ratio=1)
format_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/filtered_data/bad_format_examples_2.jsonl", split_ratio=1)
format_train_data2 = [] # There was too much data of this type, so we will not use it in the training

# USING THE DATA WITH COMET SCORES (so that the training data can be split into the two curriculum stages)
choose_train_data0, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/curriculum_data/choose_examples_0.jsonl") # Using _0 file because it has bigger requirement for the comet score difference then the old version
choose_train_data1, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/curriculum_data/choose_examples_1.jsonl", split_ratio=1)
choose_train_data2, _ = load_train_val_data("/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data/curriculum_data/choose_examples_2.jsonl", split_ratio=1)


# Merging the training data from all files into the appropriate 4 types
lang_train_data = lang_train_data0 + lang_train_data1 + lang_train_data2
short_train_data = short_train_data0 + short_train_data1 + short_train_data2
choose_train_data = choose_train_data0 + choose_train_data1 + choose_train_data2
format_train_data = format_train_data0 + format_train_data1 + format_train_data2


if should_print: print("[load_data_curriculum.py]: Training data of type 'bad_lang_examples':   ", len(lang_train_data))
if should_print: print("[load_data_curriculum.py]: Training data of type 'short_examples':      ", len(short_train_data))
if should_print: print("[load_data_curriculum.py]: Training data of type 'choose_examples':     ", len(choose_train_data))
if should_print: print("[load_data_curriculum.py]: Training data of type 'bad_format_examples': ", len(format_train_data))
if should_print: print("[load_data_curriculum.py]: *" * 50)


def clean_data(data):
    for example in data:
        example["prompt"] = [{"role": "user", "content": example["prompt"].replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model", "")}]
        example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
        example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]

    # remove all fields except "prompt", "chosen" and "rejected"
    for example in data:
        example.pop("src", None)
        example.pop("chosen_score", None)
        example.pop("rejected_score", None)

    return data


# ------------------------ MEDIAN OF CHOOSE_DATA -------------------------
# if should_print: print("[load_data_curriculum.py]: fields in choose_train_data:", choose_train_data[0].keys())

# Sort choose_train_data by the difference between chosen and rejected scores
choose_train_data = sorted(choose_train_data, key=lambda x: x['chosen_score'] - x['rejected_score'], reverse=True)

# get index of median in the list sorted by (chosen_score - rejected_score)
median_index = len(choose_train_data) // 2
# if should_print: print("[load_data_curriculum.py]: Median difference between chosen and rejected scores:", choose_train_data[median_index]['chosen_score'] - choose_train_data[median_index]['rejected_score'])
# -------------------------------------------------------------------------



# -------------------------- EVALUATION DATA ----------------------------
val_data = clean_data(lang_val_data + short_val_data + choose_val_data)
if should_print: print("[load_data_curriculum.py]: Evaluation data size:", len(val_data))
# -------------------------------------------------------------------------

# -------------------------- CURRICULUM STAGE 0 --------------------------
curriculum_0_train_data = clean_data(lang_train_data + short_train_data + format_train_data)
# take only first 70% of the data
curriculum_0_train_data = curriculum_0_train_data[:int(len(curriculum_0_train_data) * 2/3)]
if should_print: print("[load_data_curriculum.py]: Curriculum stage 0 training data size:", len(curriculum_0_train_data))
# -------------------------------------------------------------------------

# -------------------------- CURRICULUM STAGE 1 --------------------------
curriculum_1_train_data = clean_data(choose_train_data[:median_index])  # Use the first half of the sorted data
if should_print: print("[load_data_curriculum.py]: Curriculum stage 1 training data size:", len(curriculum_1_train_data))
# -------------------------------------------------------------------------

# -------------------------- CURRICULUM STAGE 2 --------------------------
curriculum_2_train_data = clean_data(choose_train_data[median_index:])  # Use the second half of the sorted data
if should_print: print("[load_data_curriculum.py]: Curriculum stage 2 training data size:", len(curriculum_2_train_data))
# -------------------------------------------------------------------------



def get_train_data(curriculum_stage=0):
    """
    Returns the training data for the specified curriculum stage.
    :param curriculum_stage: 0, 1 or 2
    :return: list of training data
    """
    if curriculum_stage == 0:
        return curriculum_0_train_data
    elif curriculum_stage == 1:
        return curriculum_1_train_data
    elif curriculum_stage == 2:
        return curriculum_2_train_data
    else:
        raise ValueError("curriculum_stage must be 0, 1 or 2")


# if should_print:
#     train_data = get_train_data(0)
#     print("First element of curriculum stage 0 training data:", train_data[0]['prompt'])
#     print("Middle element of curriculum stage 0 training data:", train_data[len(train_data) // 2]['prompt'])
#     print("Last element of curriculum stage 0 training data:", train_data[-1]['prompt'])

#     print("_" * 50)

#     print("First element of curriculum stage 1 training data:", get_train_data(1)[0]['prompt'])
#     print("Middle element of curriculum stage 1 training data:", get_train_data(1)[len(get_train_data(1)) // 2]['prompt'])
#     print("Last element of curriculum stage 1 training data:", get_train_data(1)[-1]['prompt'])

#     print("_" * 50)

#     print("First element of curriculum stage 2 training data:", get_train_data(2)[0]['prompt'])
#     print("Middle element of curriculum stage 2 training data:", get_train_data(2)[len(get_train_data(2)) // 2]['prompt'])
#     print("Last element of curriculum stage 2 training data:", get_train_data(2)[-1]['prompt'])
    
#     print("_" * 50)

#     print("First element of evaluation data:", val_data[0]['prompt'])
#     print("Middle element of evaluation data:", val_data[len(val_data) // 2]['prompt'])
#     print("Last element of evaluation data:", val_data[-1]['prompt'])
