from datasets import load_dataset
import json
import pandas as pd
import random

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
    GOOD_SCORE = 0.86

    BAD_STARTS = [ "Slovenian translation:", "#Slovene translation:", "**Slovene translation:**", "Translation:", "Prevod:", "Prevod v slovenščino:", "Slovenski prevod:", "**Slovenski prevod:**", "Prevod besedila v slovenščino",  "Prevod v slovenščino:", "Slovenski prevod:", "**Slovenski prevod:**", "Prevod besedila v slovenščino"]
    def add_bad_start(text):
        random_bad_start = random.choice(BAD_STARTS)
        return f"{random_bad_start}\n{text}"

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
    gams_data = data[data["comet_score_gams"] >= GOOD_SCORE]
    eurollm_data = data[data["comet_score_eurollm"] >= GOOD_SCORE]

    print("Shape of the gams_data data:", gams_data.shape)
    print("Shape of the eurollm_data data:", eurollm_data.shape)
    print("-"*50)

    # Preference dataset (gams)
    # make each sl_translation_eurollm now be equal to the appropriate sl_translation_gams with a bad start
    gams_data["sl_translation_eurollm"] = gams_data["sl_translation_gams"].apply(add_bad_start)
    gams_data = gams_data.drop(columns=["id", "language_eurollm", "language_gams", "text", "title", "url", "Prompt_eurollm", "relative_length_gams", "relative_length_eurollm"])
    gams_data = gams_data.rename(columns={"Prompt_gams": "prompt", "sl_translation_gams": "chosen", "sl_translation_eurollm": "rejected"})
    gams_data = gams_data.rename(columns={"comet_score_gams": "chosen_score", "comet_score_eurollm": "rejected_score"})
    gams_data['src'] = "gams"

    print("Shape of the gams_data data after formatting:", gams_data.shape)
    for column in gams_data.columns:
        print(f" - {column}")

    # Preference dataset (eurollm)
    # make each sl_translation_gams now be equal to the appropriate sl_translation_eurollm with a bad start
    eurollm_data["sl_translation_gams"] = eurollm_data["sl_translation_eurollm"].apply(add_bad_start)
    eurollm_data = eurollm_data.drop(columns=["id", "language_eurollm", "language_gams", "text", "title", "url", "Prompt_eurollm", "relative_length_gams", "relative_length_eurollm"])
    eurollm_data = eurollm_data.rename(columns={"Prompt_gams": "prompt", "sl_translation_eurollm": "chosen", "sl_translation_gams": "rejected"})
    eurollm_data = eurollm_data.rename(columns={"comet_score_eurollm": "chosen_score", "comet_score_gams": "rejected_score"})
    eurollm_data['src'] = "eurollm"

    print("Shape of the eurollm_data data after formatting:", eurollm_data.shape)
    for column in eurollm_data.columns:
        print(f" - {column}")

    # Merge the two dataframes
    bad_format_data = pd.concat([gams_data, eurollm_data], ignore_index=True)
    # print("-"*50)
    # print("Shape of the bad_format_data data:", bad_format_data.shape)
    # for column in bad_format_data.columns:
    #     print(f" - {column}")

    if ADAPT_TO_NEMO:
        # Adapt the data to the NeMo format
        bad_format_data = addapt_to_nemo(bad_format_data)
        print("Shape of the bad_format_data data after adaptation:", bad_format_data.shape)
        for column in bad_format_data.columns:
            print(f" - {column}")

    # Save the data
    bad_format_data.to_json("bad_format_examples.jsonl", orient="records", lines=True, force_ascii=False)

if __name__=="__main__":
    main()
