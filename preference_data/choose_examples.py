from datasets import load_dataset
import json
import pandas as pd

ADAPT_TO_NEMO = False

def addapt_to_nemo(data):
    for example in data:
        example["prompt"] = [{"role": "user", "content": example["prompt"].replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model", "")}]
        example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
        example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]
    data = data.rename(columns={"chosen": "chosen_response", "rejected": "rejected_response"})
    return data

def main():
    # USIN THIS AS MINIMUM SCORE DIFFERENCE TO CONSIDER A PREFERENCE
    SCORE_THRESHOLD = 0.03

    data = pd.read_json("../language_identification/paired_data_with_scores.jsonl", orient="records", lines=True)
    print("Shape of the data:", data.shape)

    # Filtering out the rows where both translations are in Slovene
    data = data[(data["language_eurollm"] == "SL") & (data["language_gams"] == "SL")]
    print("Shape of the data with both translations in Slovene:", data.shape)

    # Filtering out the rows where some translations are too short
    SIZE_THRESHOLD = 0.7
    data = data[(data["relative_length_gams"] > SIZE_THRESHOLD) & (data["relative_length_eurollm"] > SIZE_THRESHOLD)]
    print("Shape of the data with translations longer than the threshold:", data.shape)
    print("-"*50)

    # Dividing the data into two dataframes based on the comet scores
    gams_better = data[data["comet_score_gams"] >= data["comet_score_eurollm"] + SCORE_THRESHOLD]
    eurollm_better = data[data["comet_score_eurollm"] > data["comet_score_gams"] + SCORE_THRESHOLD]

    print("Shape of the gams_better data:", gams_better.shape)
    print("Shape of the eurollm_better data:", eurollm_better.shape)
    print("-"*50)

    # Preference dataset (gams)
    gams_better = gams_better.drop(columns=["id", "language_eurollm", "language_gams", "text", "title", "url", "Prompt_eurollm", "relative_length_gams", "relative_length_eurollm"])
    gams_better = gams_better.rename(columns={"Prompt_gams": "prompt", "sl_translation_gams": "chosen", "sl_translation_eurollm": "rejected"})
    gams_better = gams_better.rename(columns={"comet_score_gams": "chosen_score", "comet_score_eurollm": "rejected_score"})
    gams_better['src'] = "gams"

    print("Shape of the gams_better data after formatting:", gams_better.shape)
    for column in gams_better.columns:
        print(f" - {column}")

    # Preference dataset (eurollm)
    eurollm_better = eurollm_better.drop(columns=["id", "language_eurollm", "language_gams", "text", "title", "url", "Prompt_eurollm", "relative_length_gams", "relative_length_eurollm"])
    eurollm_better = eurollm_better.rename(columns={"Prompt_gams": "prompt", "sl_translation_eurollm": "chosen", "sl_translation_gams": "rejected"})
    eurollm_better = eurollm_better.rename(columns={"comet_score_eurollm": "chosen_score", "comet_score_gams": "rejected_score"})
    eurollm_better['src'] = "eurollm"

    print("Shape of the eurollm_better data after formatting:", eurollm_better.shape)
    for column in eurollm_better.columns:
        print(f" - {column}")

    # Merge the two dataframes
    preference_data = pd.concat([gams_better, eurollm_better], ignore_index=True)
    # print("-"*50)
    # print("Shape of the preference_data data:", preference_data.shape)
    # for column in preference_data.columns:
    #     print(f" - {column}")

    if ADAPT_TO_NEMO:
        # Adapt the data to the NeMo format
        preference_data = addapt_to_nemo(preference_data)
        print("Shape of the preference_data data after adaptation:", preference_data.shape)
        for column in preference_data.columns:
            print(f" - {column}")

    # Save the data
    preference_data.to_json("choose_examples.jsonl", orient="records", lines=True, force_ascii=False)

if __name__=="__main__":
    main()
