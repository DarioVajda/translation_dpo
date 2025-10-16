import json
import os
from tqdm import tqdm

# Load the GAMS translations from /shared/workspace/povejmo/translation_optimization/wiki_eval/comet_scores/eurollm_translations.jsonl
gams_list = []
with open("/workspace/data_pipeline/gams/language_id/SL/all_translation_0.jsonl", "r") as file:
    for line in tqdm(file, desc="Loading GAMS translations"):
        gams_list.append(json.loads(line.strip()))

# load the Eurollm translations from /shared/workspace/povejmo/translation_optimization/wiki_eval/comet_scores/eurollm_translations.jsonl
eurollm_list = []
with open("/workspace/data_pipeline/eurollm/language_id/SL/all_translation_0.jsonl", "r") as file:
    for line in tqdm(file, desc="Loading Eurollm translations"):
        eurollm_list.append(json.loads(line.strip()))

# load the GAMS DPO translations from /shared/workspace/povejmo/translation_optimization/wiki_eval/comet_scores/gams_dpo_translations.jsonl
gams_dpo_list = []
with open("/workspace/data_pipeline/translator_v2/language_id/SL/all_translation_0.jsonl", "r") as file:
    for line in tqdm(file, desc="Loading GAMS DPO translations"):
        gams_dpo_list.append(json.loads(line.strip()))

print("Number of GAMS translations:", len(gams_list))
print("Number of Eurollm translations:", len(eurollm_list))
print("Number of GAMS DPO translations:", len(gams_dpo_list))

# Calculate average of all comet scores for each model
def calculate_average_scores(translations, model_name):
    total_score = 0
    count = 0
    for example in translations:
        if "comet_score" in example:
            total_score += example["comet_score"]
            count += 1
    average_score = total_score / count if count > 0 else 0
    print(f"{model_name}: {average_score:.4f}")

print("-" * 30)
print("BOTH DATASETS")
print("-" * 30)
calculate_average_scores(gams_list, "GAMS")
calculate_average_scores(eurollm_list, "EuroLLM")
calculate_average_scores(gams_dpo_list, "GAMS DPO")

# Calculate the average comet score for ccnews articles (detect them by checking if the first character in id is 'h')
print("-" * 30)
print("CCNEWS ARTICLES")
print("-" * 30)
calculate_average_scores([example for example in gams_list if example["id"].startswith("h")], "GAMS")
calculate_average_scores([example for example in eurollm_list if example["id"].startswith("h")], "EuroLLM")
calculate_average_scores([example for example in gams_dpo_list if example["id"].startswith("h")], "GAMS DPO")

# Calculate the average comet score for wikipedia articles (detect them by checking if the first character in id is not 'h')
print("-" * 30)
print("WIKIPEDIA ARTICLES")
print("-" * 30)
calculate_average_scores([example for example in gams_list if not example["id"].startswith("h")], "GAMS")
calculate_average_scores([example for example in eurollm_list if not example["id"].startswith("h")], "EuroLLM")
calculate_average_scores([example for example in gams_dpo_list if not example["id"].startswith("h")], "GAMS DPO")
