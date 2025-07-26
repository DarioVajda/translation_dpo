import json
import pandas as pd
from argparse import ArgumentParser
import random
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
    GOOD_SCORE = 0.86

    BAD_STARTS = [ "Slovenian translation:", "#Slovene translation:", "**Slovene translation:**", "Translation:", "Prevod:", "Prevod v slovenščino:", "Slovenski prevod:", "**Slovenski prevod:**", "Prevod besedila v slovenščino",  "Prevod v slovenščino:", "Slovenski prevod:", "**Slovenski prevod:**", "Prevod besedila v slovenščino"]
    def add_bad_start(text):
        random_bad_start = random.choice(BAD_STARTS)
        return f"{random_bad_start}\n{text}"

    # path = f"../comet_score/scored_data/ccnews_2{f'_{id}' if id>0 else ''}.jsonl"
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
    gams_data = data[data["comet_score_1"] >= GOOD_SCORE]
    eurollm_data = data[data["comet_score_2"] >= GOOD_SCORE]

    print("Shape of the gams_data data:", gams_data.shape)
    print("Shape of the eurollm_data data:", eurollm_data.shape)
    print("-"*50)

    # Preference dataset (gams)
    # make each sl_translation_2 now be equal to the appropriate sl_translation_1 with a bad start
    gams_data["sl_translation_2"] = gams_data["sl_translation_1"].apply(add_bad_start)
    gams_data = gams_data.drop(columns=["id", "language_2", "language_1", "text", "title", "url", "Prompt_2", "relative_length_1", "relative_length_2"])
    gams_data = gams_data.rename(columns={"Prompt_1": "prompt", "sl_translation_1": "chosen", "sl_translation_2": "rejected"})
    gams_data = gams_data.rename(columns={"comet_score_1": "chosen_score", "comet_score_2": "rejected_score"})
    gams_data['src'] = "gams"

    print("Shape of the gams_data data after formatting:", gams_data.shape)
    for column in gams_data.columns:
        print(f" - {column}")

    # Preference dataset (eurollm)
    # make each sl_translation_1 now be equal to the appropriate sl_translation_2 with a bad start
    eurollm_data["sl_translation_1"] = eurollm_data["sl_translation_2"].apply(add_bad_start)
    eurollm_data = eurollm_data.drop(columns=["id", "language_2", "language_1", "text", "title", "url", "Prompt_2", "relative_length_1", "relative_length_2"])
    eurollm_data = eurollm_data.rename(columns={"Prompt_1": "prompt", "sl_translation_2": "chosen", "sl_translation_1": "rejected"})
    eurollm_data = eurollm_data.rename(columns={"comet_score_2": "chosen_score", "comet_score_1": "rejected_score"})
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
    bad_format_data.to_json(output_path, orient="records", lines=True, force_ascii=False)

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
