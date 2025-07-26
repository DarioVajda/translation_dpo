import json
import pandas as pd
from argparse import ArgumentParser

ADAPT_TO_NEMO = False

def addapt_to_nemo(data):
    for example in data:
        example["prompt"] = [{"role": "user", "content": example["prompt"].replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model", "")}]
        example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
        example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]
    data = data.rename(columns={"chosen": "chosen_response", "rejected": "rejected_response"})
    return data

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--paired_data_path",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSONL file."
    )
    return parser.parse_args()

def main(id=0, path=None):
    args=parse_args()
    input_path = args.paired_data_path
    output_path = args.output_path

    # USIN THIS AS MINIMUM SCORE DIFFERENCE TO CONSIDER A PREFERENCE
    SCORE_THRESHOLD = 0.05

    # data = pd.read_json(f"../language_identification/paired_data_with_scores{f'_{id}' if id>0 else ''}.jsonl", orient="records", lines=True)
    data = pd.read_json(input_path, orient="records", lines=True)
    print("Shape of the data:", data.shape)

    # Filtering out the rows where both translations are in Slovene
    data = data[(data["language_2"] == "SL") & (data["language_1"] == "SL")]
    print("Shape of the data with both translations in Slovene:", data.shape)

    # Filtering out the rows where some translations are too short
    SIZE_THRESHOLD = 0.7
    data = data[(data["relative_length_1"] > SIZE_THRESHOLD) & (data["relative_length_2"] > SIZE_THRESHOLD)]
    print("Shape of the data with translations longer than the threshold:", data.shape)
    print("-"*50)

    # Dividing the data into two dataframes based on the comet scores
    gams_better = data[data["comet_score_1"] >= data["comet_score_2"] + SCORE_THRESHOLD]
    eurollm_better = data[data["comet_score_2"] > data["comet_score_1"] + SCORE_THRESHOLD]

    print("Shape of the gams_better data:", gams_better.shape)
    print("Shape of the eurollm_better data:", eurollm_better.shape)
    print("-"*50)

    # Preference dataset (gams)
    gams_better = gams_better.drop(columns=["id", "language_2", "language_1", "text", "title", "url", "Prompt_2", "relative_length_1", "relative_length_2"])
    gams_better = gams_better.rename(columns={"Prompt_1": "prompt", "sl_translation_1": "chosen", "sl_translation_2": "rejected"})
    gams_better = gams_better.rename(columns={"comet_score_1": "chosen_score", "comet_score_2": "rejected_score"})
    gams_better['src'] = "gams"

    print("Shape of the gams_better data after formatting:", gams_better.shape)
    for column in gams_better.columns:
        print(f" - {column}")

    # Preference dataset (eurollm)
    eurollm_better = eurollm_better.drop(columns=["id", "language_2", "language_1", "text", "title", "url", "Prompt_2", "relative_length_1", "relative_length_2"])
    eurollm_better = eurollm_better.rename(columns={"Prompt_1": "prompt", "sl_translation_2": "chosen", "sl_translation_1": "rejected"})
    eurollm_better = eurollm_better.rename(columns={"comet_score_2": "chosen_score", "comet_score_1": "rejected_score"})
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
    preference_data.to_json(output_path, orient="records", lines=True, force_ascii=False)

if __name__=="__main__":
    main()
    # for id in [1, 2]:
    #     print('*'*60)
    #     print(f"Processing data with id {id}")
    #     print('*'*60)
    #     main(id)

    # paths = [
    #     ("../language_identification/ccnews_paired/1.jsonl", 1),
    #     ("../language_identification/ccnews_paired/2.jsonl", 2),
    # ]

    # for (path, id) in paths:
    #     print('*'*60)
    #     print(f"Processing data from path {path}")
    #     print('*'*60)
    #     main(id=id, path=path)