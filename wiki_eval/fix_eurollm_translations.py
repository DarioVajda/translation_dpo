import os
import json



# load ./all_translations.jsonl
all_translations = []
with open("./all_translations.jsonl", "r") as file:
    for line in file:
        all_translations.append(json.loads(line.strip()))


# load ./ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_eurollm9b_translation_299.jsonl
eurollm_translations = []
with open("/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_eurollm9b_translation_299.jsonl", "r") as file:
    for line in file:
        eurollm_translations.append(json.loads(line.strip()))


for example in all_translations:
    eurollm_translation_index = [ i for i in range(len(eurollm_translations)) if eurollm_translations[i]["id"] == example["id"] ][0]
    example["eurollm_translation"] = eurollm_translations[eurollm_translation_index]["sl_translation"]

# save the updated all_translations to ./all_translations_fixed.jsonl and don't force ASCII encoding
with open("./all_translations_fixed.jsonl", "w") as file:
    for example in all_translations:
        file.write(json.dumps(example, ensure_ascii=False) + "\n")