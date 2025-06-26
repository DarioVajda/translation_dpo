import json
import os

# load the /ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/gams_translations.jsonl file
gams = []
with open("/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/language_id/gams_translations.jsonl", "r") as file:
    for line in file:
        gams.append(json.loads(line.strip()))

# load the /ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/language_id/eurollm_translations.jsonl file
eurollm = []
with open("/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/language_id/eurollm_translations.jsonl", "r") as file:
    for line in file:
        eurollm.append(json.loads(line.strip()))

# load the /ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/language_id/gams_dpo_translations.jsonl file
gams_dpo = []
with open("/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/language_id/gams_dpo_translations.jsonl", "r") as file:
    for line in file:
        gams_dpo.append(json.loads(line.strip()))

print("Number of GAMS translations:", len(gams))
print("Number of Eurollm translations:", len(eurollm))
print("Number of GAMS DPO translations:", len(gams_dpo))


total = 0
gams_filtered = []
gams_dpo_filtered = []
eurollm_filtered = []
for gams_e in gams:
    if gams_e["lang"] != "SL":
        continue
    
    for i in range(len(eurollm)):
        if eurollm[i]["id"] == gams_e["id"]:
            eurollm_i = i
            break

    for i in range(len(gams_dpo)):
        if gams_dpo[i]["id"] == gams_e["id"]:
            gams_dpo_i = i
            break

    if eurollm[eurollm_i]['lang'] != "SL":
        continue
    if gams_dpo[gams_dpo_i]['lang'] != "SL":
        continue

    total += 1

    gams_filtered.append(gams_e)
    gams_dpo_filtered.append(gams_dpo[gams_dpo_i])
    eurollm_filtered.append(eurollm[eurollm_i])

# save the filtered datasets to JSONL files
with open("/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/intersection/gams_translations_filtered.jsonl", "w") as file:
    for example in gams_filtered:
        file.write(json.dumps(example, ensure_ascii=False) + "\n")
with open("/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/intersection/gams_dpo_translations_filtered.jsonl", "w") as file:
    for example in gams_dpo_filtered:
        file.write(json.dumps(example, ensure_ascii=False) + "\n")
with open("/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/intersection/eurollm_translations_filtered.jsonl", "w") as file:
    for example in eurollm_filtered:
        file.write(json.dumps(example, ensure_ascii=False) + "\n")