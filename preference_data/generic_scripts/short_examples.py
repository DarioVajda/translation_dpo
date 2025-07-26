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

    # Load the data
    # data = pd.read_json(f"../language_identification/paired_data_with_scores{f'_{id}' if id>0 else ''}.jsonl", orient="records", lines=True)
    data = pd.read_json(input_path, orient="records", lines=True)
    print("Shape of all data:", data.shape)

    # Using only the examples where both translations are in Slovene
    data = data[(data["language_2"] == "SL") & (data["language_1"] == "SL")]

    print("Shape of the data with both translations in Slovene:", data.shape)

    GOOD_SIZE_THRESHOLD = 0.7
    BAD_SIZE_THRESHOLD = 0.5

    gams_better = data[(data["relative_length_1"] > GOOD_SIZE_THRESHOLD) & (data["relative_length_2"] <= BAD_SIZE_THRESHOLD)]
    eurollm_better = data[(data["relative_length_2"] > GOOD_SIZE_THRESHOLD) & (data["relative_length_1"] <= BAD_SIZE_THRESHOLD)]

    # print("Shape of the gams_better data:", gams_better.shape)
    # print("Shape of the eurollm_better data:", eurollm_better.shape)
    # print("Shape of the both_bad data:", both_bac.shape)

    # Preference dataset (gams)
    gams_better = gams_better.drop(columns=["id", "language_2", "language_1", "text", "title", "url", "Prompt_2", "comet_score_1", "comet_score_2"])
    gams_better = gams_better.rename(columns={"Prompt_1": "prompt", "sl_translation_1": "chosen", "sl_translation_2": "rejected"})
    gams_better = gams_better.drop(columns=["relative_length_1", "relative_length_2"])
    gams_better['src'] = "gams"

    # Preference dataset (eurollm)
    eurollm_better = eurollm_better.drop(columns=["id", "language_2", "language_1", "text", "title", "url", "Prompt_2", "comet_score_1", "comet_score_2"])
    eurollm_better = eurollm_better.rename(columns={"Prompt_1": "prompt", "sl_translation_2": "chosen", "sl_translation_1": "rejected"})
    eurollm_better = eurollm_better.drop(columns=["relative_length_1", "relative_length_2"])
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
    short_examples.to_json(output_path, orient="records", lines=True, force_ascii=False)

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
