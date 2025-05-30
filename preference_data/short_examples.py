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

# HERE I WILL MAKE EXAMPLES OF THE DATASET WHERE
# CHOSEN IS OK AND REJECTED IS TOO SHORT
def main():
    # Load the data
    data = pd.read_json("../language_identification/paired_data.jsonl", orient="records", lines=True)
    print("Shape of all data:", data.shape)

    # Using only the examples where both translations are in Slovene
    data = data[(data["language_eurollm"] == "SL") & (data["language_gams"] == "SL")]

    print("Shape of the data with both translations in Slovene:", data.shape)

    GOOD_SIZE_THRESHOLD = 0.7
    BAD_SIZE_THRESHOLD = 0.5

    gams_better = data[(data["relative_length_gams"] > GOOD_SIZE_THRESHOLD) & (data["relative_length_eurollm"] <= BAD_SIZE_THRESHOLD)]
    eurollm_better = data[(data["relative_length_eurollm"] > GOOD_SIZE_THRESHOLD) & (data["relative_length_gams"] <= BAD_SIZE_THRESHOLD)]

    # print("Shape of the gams_better data:", gams_better.shape)
    # print("Shape of the eurollm_better data:", eurollm_better.shape)
    # print("Shape of the both_bad data:", both_bac.shape)

    # Preference dataset (gams)
    gams_better = gams_better.drop(columns=["id", "language_eurollm", "language_gams", "text", "title", "url", "Prompt_eurollm"])
    gams_better = gams_better.rename(columns={"Prompt_gams": "prompt", "sl_translation_gams": "chosen", "sl_translation_eurollm": "rejected"})
    gams_better = gams_better.drop(columns=["relative_length_gams", "relative_length_eurollm"])
    gams_better['src'] = "gams"

    # Preference dataset (eurollm)
    eurollm_better = eurollm_better.drop(columns=["id", "language_eurollm", "language_gams", "text", "title", "url", "Prompt_eurollm"])
    eurollm_better = eurollm_better.rename(columns={"Prompt_gams": "prompt", "sl_translation_eurollm": "chosen", "sl_translation_gams": "rejected"})
    eurollm_better = eurollm_better.drop(columns=["relative_length_gams", "relative_length_eurollm"])
    eurollm_better['src'] = "eurollm"

    print("Shape of the gams_better data:", gams_better.shape)
    for column in gams_better.columns:
        print(f" - {column}")
    print("Shape of the eurollm_better data:", eurollm_better.shape)
    for column in eurollm_better.columns:
        print(f" - {column}")

    # Merge the two dataframes
    short_examples = pd.concat([gams_better, eurollm_better], ignore_index=True)
    print("Shape of the short_examples data:", short_examples.shape)

    if ADAPT_TO_NEMO:
        # Adapt the data to the NeMo format
        short_examples = addapt_to_nemo(short_examples)
        print("Shape of the short_examples data after adaptation:", short_examples.shape)
        for column in short_examples.columns:
            print(f" - {column}")

    # Save the data
    short_examples.to_json("short_examples.jsonl", orient="records", lines=True, force_ascii=False)

if __name__=="__main__":
    main()
