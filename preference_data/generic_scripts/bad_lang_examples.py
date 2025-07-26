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
    # Check the shape of the data
    print("Shape of the data:", data.shape)

    # I want to save the rows of data where exactly one of the columns language_2 or language_1 is "SL" and the other is something else
    gams_sl = data[(data["language_2"] != "SL") & (data["language_1"] == "SL")]
    eurollm_sl = data[(data["language_2"] == "SL") & (data["language_1"] != "SL")]
    both_bad = data[(data["language_2"] != "SL") & (data["language_1"] != "SL")]

    SIZE_THRESHOLD = 0.7

    # Preference dataset (gams)
    gams_sl = gams_sl.drop(columns=["id", "language_2", "language_1", "text", "title", "url", "Prompt_2", "comet_score_1", "comet_score_2"])
    gams_sl = gams_sl.rename(columns={"Prompt_1": "prompt", "sl_translation_1": "chosen", "sl_translation_2": "rejected"})
    gams_sl = gams_sl[gams_sl["relative_length_1"] > SIZE_THRESHOLD]
    gams_sl = gams_sl.drop(columns=["relative_length_1", "relative_length_2"])
    gams_sl['src'] = "gams"

    # Preference dataset (eurollm)
    eurollm_sl = eurollm_sl.drop(columns=["id", "language_2", "language_1", "text", "title", "url", "Prompt_2", "comet_score_1", "comet_score_2"])
    eurollm_sl = eurollm_sl.rename(columns={"Prompt_1": "prompt", "sl_translation_2": "chosen", "sl_translation_1": "rejected"})
    eurollm_sl = eurollm_sl[eurollm_sl["relative_length_2"] > SIZE_THRESHOLD]
    eurollm_sl = eurollm_sl.drop(columns=["relative_length_1", "relative_length_2"])
    eurollm_sl['src'] = "eurollm"

    print("Shape of the gams_sl data:", gams_sl.shape)
    for column in gams_sl.columns:
        print(f" - {column}")
    print("Shape of the eurollm_sl data:", eurollm_sl.shape)
    for column in eurollm_sl.columns:
        print(f" - {column}")

    # Merge the two dataframes
    bad_lang_examples = pd.concat([gams_sl, eurollm_sl], ignore_index=True)
    print("Shape of the bad_lang_examples data:", bad_lang_examples.shape)

    if ADAPT_TO_NEMO:
        # Adapt the data to the NeMo format
        bad_lang_examples = addapt_to_nemo(bad_lang_examples)
        print("Shape of the bad_lang_examples data after adaptation:", bad_lang_examples.shape)
        for column in bad_lang_examples.columns:
            print(f" - {column}")

    # Save the data
    bad_lang_examples.to_json(output_path, orient="records", lines=True, force_ascii=False)

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