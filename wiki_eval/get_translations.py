print("Starting the script...")
from transformers import pipeline, AutoTokenizer
print("imported pipeline from transformers")
import torch
print("imported torch")
from tqdm import tqdm
print("imported tqdm")

import json
print("imported json")

from datasets import load_from_disk
print("imported load_from_disk from datasets")

# model_id = "cjvt/GaMS-9B-Instruct"
# model_id = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct-translate-v2"
model_id = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct-translate-v3"
# model_id = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct-translate-v4"
# model_id = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/trained_models/Curriculum_DPO_models/GaMS-9B-DPO-Curri-0"
# model_id = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/trained_models/Curriculum_DPO_models/GaMS-9B-DPO-Curri-2"

device_id = 3

pline = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    # device=device_id,
)
print("Initialized pipeline with model:", model_id)

def fixed_selection(n, m, id, seed=42):
    return [ i for i in range(n) if i % 300 == id ]

gams_paths = [
    "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_gams9b_translation_299.jsonl"
]
eurollm_paths = [
    "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_eurollm9b_translation_299.jsonl"
]

# Load the dataset from a JSONL file
gams_list = []
for gams_path in gams_paths:
    with open(gams_path, "r") as file:
        for line in file:
            obj = json.loads(line.strip())
            if not obj["id"]: obj["id"] = obj["url"]
            gams_list.append(obj)
eurollm_list = []
for eurollm_path in eurollm_paths:
    with open(gams_path, "r") as file:
        for line in file:
            obj = json.loads(line.strip())
            if not obj["id"]: obj["id"] = obj["url"]
            gams_list.append(obj)

# {"id", "url", "title", "text", "Prompt", "Problematic", "sl_translation"}

paired_list = []
count = 0
for gams_example in gams_list:
    # find index of element with same id
    eurollm_example = [ e for e in eurollm_list if e["id"] == gams_example["id"] ][0]
    if len(gams_example["text"]) > 3000: continue
    count += 1
    # if count < device_id * 100: continue
    paired_list.append({
        "id": gams_example["id"],
        "url": gams_example["url"],
        "title": gams_example["title"],
        "text": gams_example["text"],
        "Prompt_gams": gams_example["Prompt"],
        "Prompt_eurollm": eurollm_example["Prompt"],
        "Problematic_gams": gams_example["Problematic"],
        "Problematic_eurollm": eurollm_example["Problematic"],
        "gams_translation": gams_example["sl_translation"],
        "eurollm_translation": eurollm_example["sl_translation"],
    })
    # if len(paired_list) == 100:
    #     break

print("There are {} entries in the evaluation dataset.".format(len(paired_list)))

# Load a bunch of wikipedia articles for translation
def get_messages():
    prompts = [ [{
        "role": "user",
        "content": f"Prevedi naslednje angleško besedilo v slovenščino.\n# {example['title']}\n\n{example['text']}"
    }, example] for example in paired_list ]
    return prompts

messages = get_messages()
# print(f"Number of messages to translate: {len(messages)}")
# print("First message:", messages[0][0]["content"])
# print("second message:", messages[1][0]["content"])

translation_list = []

# # Iterate over the messages and generate translations
for message in tqdm(messages, desc="Translating"):
    response = pline([message[0]], max_new_tokens=2048)
    
    prompt = message[0]["content"]
    res = response[0]["generated_text"][-1]["content"]
    translation_object = message[1]
    translation_object["gams_dpo_translation"] = res
    translation_list.append(translation_object)

# Save the translations to a file
output_file_path = f"/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/gams_dpo_translations_{device_id}.jsonl"
with open(output_file_path, "w") as output_file:
    for translation in translation_list:
        translation_object_text = json.dumps(translation, ensure_ascii=False)
        output_file.write(translation_object_text + "\n")

print(f"Translations saved to {output_file_path}")