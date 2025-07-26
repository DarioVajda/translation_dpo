print("Starting the script...")
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
print("imported pipeline from transformers")
import torch
print("imported torch")
from tqdm import tqdm
print("imported tqdm")

import json
print("imported json")

from datasets import load_from_disk
print("imported load_from_disk from datasets")

torch.set_float32_matmul_precision("high")


gams_paths = [
    # "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_gams9b_translation_299.jsonl",
    # "/workspace/get_translations/translations/gams_ccnews_0.jsonl",
    "/workspace/get_translations/translations/gams_wiki_eval.jsonl"
]
eurollm_paths = [
    # "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_eurollm9b_translation_299.jsonl",
    # "/workspace/get_translations/translations/eurollm_ccnews_0.jsonl",
    "/workspace/get_translations/translations/eurollm_wiki_eval.jsonl"
]

def format_object(obj):
    if not ('id' in obj):
        obj["id"] = obj["requested_url"]
    if not ('text' in obj):
        obj["text"] = obj["plain_text"]
        del obj["plain_text"]
    if not ('url' in obj):
        obj["url"] = obj["requested_url"]
    return obj

# Load the dataset from a JSONL file
gams_list = []
for gams_path in gams_paths:
    with open(gams_path, "r") as file:
        for line in file:
            obj = json.loads(line.strip())
            gams_list.append(format_object(obj))
eurollm_list = []
for eurollm_path in eurollm_paths:
    with open(eurollm_path, "r") as file:
        for line in file:
            obj = json.loads(line.strip())
            eurollm_list.append(format_object(obj))


# {"id", "url", "title", "text", "Prompt", "Problematic", "sl_translation"}

# print(eurollm_list[0])
# assert True, "Breakpoint"

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
# messages = messages[:20]
# print(f"Number of messages to translate: {len(messages)}")
# print("First message:", messages[0][0]["content"])
# print("second message:", messages[1][0]["content"])


# model_id = "DarioVajda/GaMS-DPO-Translator"
model_id = "/povejmo/models/hf_models/GaMS-9B-SFT_Translator-grouped"
batch_size = 16              # tune this to fill your GPUs without OOM
max_new_tokens = 2048        # same as before

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",                 # auto‐shard across all GPUs
    # load_in_8bit=True,                 # compress weights to 8-bit
    torch_dtype=torch.bfloat16,        # use FP16 compute
    offload_folder="offload",          # offload CPU if needed
    offload_state_dict=True,
    low_cpu_mem_usage=True,
)

pler = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",          # again, let HF handle the GPUs
    framework="pt",
    batch_size=batch_size,
    return_full_text=False,     # only return new tokens
)
print("Initialized pipeline with model:", model_id)

translation_list = []

for i in tqdm(range(0, len(messages), batch_size), desc="Translating"):
    chunk = messages[i : i + batch_size]
    prompts = [[msg[0]] for msg in chunk]
    print(i, "/", len(messages)//batch_size)

    # generate all at once
    outputs = pler(prompts, max_new_tokens=max_new_tokens)
    
    for out_list, (_, trans_obj) in zip(outputs, chunk):
        # out_list is a list of dicts; grab the first one
        first = out_list[0]
        gen = first["generated_text"]
        trans_obj["gams_dpo_translation"] = gen
        translation_list.append(trans_obj)

# pline = pipeline(
#     "text-generation",
#     model=model_id,
#     device_map="auto",
#     # device=device_id,
# )

# translation_list = []
# # Iterate over the messages and generate translations
# for message in tqdm(messages, desc="Translating"):
#     response = pline([message[0]], max_new_tokens=2048)
    
#     prompt = message[0]["content"]
#     res = response[0]["generated_text"][-1]["content"]
#     translation_object = message[1]
#     translation_object["gams_dpo_translation"] = res
#     translation_list.append(translation_object)

# Save the translations to a file
output_file_path = f"/workspace/wiki_eval/sft_model_wiki_translations.jsonl"
with open(output_file_path, "w") as output_file:
    for translation in translation_list:
        translation_object_text = json.dumps(translation, ensure_ascii=False)
        output_file.write(translation_object_text + "\n")

print(f"Translations saved to {output_file_path}")