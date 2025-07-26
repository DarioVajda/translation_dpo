
import json
import os
from tqdm import tqdm

# open ./language_id/gams_dpo_translations.jsonl
gams_dpo_list = []
with open("./intersection/sft/gams_sft_translations_filtered.jsonl", "r") as file:
    for line in tqdm(file, desc="Loading GAMS DPO translations"):
        gams_dpo_list.append(json.loads(line.strip()))

# open ./language_id/gams_translations.jsonl
gams_list = []
with open("./intersection/sft/gams_translations_filtered.jsonl", "r") as file:
    for line in tqdm(file, desc="Loading GAMS translations"):
        gams_list.append(json.loads(line.strip()))

# open ./language_id/eurollm_translations.jsonl
eurollm_list = []
with open("./intersection/sft/eurollm_translations_filtered.jsonl", "r") as file:
    for line in tqdm(file, desc="Loading Eurollm translations"):
        eurollm_list.append(json.loads(line.strip()))




for example_list, model_name in zip([gams_dpo_list, gams_list, eurollm_list], ['GAMS SFT', 'GAMS', 'Eurollm']):
    too_short = 0
    total = 0
    for example in example_list:
        if example["lang"] != "SL":
            continue
        total += 1
        if len(example["sl_translation"]) / len(example["text"]) < 0.7:
            too_short += 1
    print(f"Number of examples with translation too short in {model_name}: {too_short} out of {len(example_list)} ({too_short / len(example_list) * 100:.2f}%)")
    print(f"Percentage of examples with translation too short in {model_name}: {too_short / total * 100:.2f}%")
    print("*" * 60)
