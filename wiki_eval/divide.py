import json

# "id", "url", "title", "text", "Prompt_gams", "Prompt_eurollm", "Problematic_gams", "Problematic_eurollm", "gams_translation", "eurollm_translation", "gams_dpo_translation"

all_translations = []
with open("./all_translations_fixed.jsonl", "r") as file:
    for line in file:
        all_translations.append(json.loads(line.strip()))


gams = [ {
    "id": example["id"],
    "url": example["url"],
    "title": example["title"],
    "text": example["text"],
    "Prompt": example["Prompt_gams"],
    "Problematic": example["Problematic_gams"],
    "sl_translation": example["gams_translation"],
} for example in all_translations ]

eurollm = [ {
    "id": example["id"],
    "url": example["url"],
    "title": example["title"],
    "text": example["text"],
    "Prompt": example["Prompt_eurollm"],
    "Problematic": example["Problematic_eurollm"],
    "sl_translation": example["eurollm_translation"],
} for example in all_translations ]

gams_dpo = [ {
    "id": example["id"],
    "url": example["url"],
    "title": example["title"],
    "text": example["text"],
    "Prompt": example["Prompt_gams"],
    "Problematic": example["Problematic_gams"],
    "sl_translation": example["gams_dpo_translation"],
} for example in all_translations ]

print("Number of GAMS translations:", len(gams))
print("Number of Eurollm translations:", len(eurollm))
print("Number of GAMS DPO translations:", len(gams_dpo))

# Save the divided datasets to JSONL files
with open("./gams_translations.jsonl", "w") as file:
    for example in gams:
        file.write(json.dumps(example) + "\n")
with open("./eurollm_translations.jsonl", "w") as file:
    for example in eurollm:
        file.write(json.dumps(example) + "\n")
with open("./gams_dpo_translations.jsonl", "w") as file:
    for example in gams_dpo:
        file.write(json.dumps(example) + "\n")