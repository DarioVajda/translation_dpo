import os
import json
import pandas as pd
from argparse import ArgumentParser

def load_multilang_json(base_dir):
    records = []
    # iterate over each language folder
    for lang in os.listdir(base_dir):
        lang_dir = os.path.join(base_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        # iterate over each file in that language folder
        for fname in os.listdir(lang_dir):
            if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                continue
            fpath = os.path.join(lang_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                # one JSON object per line
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    obj["language"] = lang
                    records.append(obj)
    return pd.DataFrame(records)
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--language_id_path1",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--language_id_path2",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--paired_data_path",
        type=str,
        required=True,
        help="Path to the output JSONL file where paired data will be saved."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for id in [1, 2]:
        dfs = {}
        print('*'*60)
        print(f"Processing data with id {id}")
        print('*'*60)

        # Load the data
        for (path, i) in [ (args.language_id_path1, '1'), (args.language_id_path2, '2') ]:
            model = i
            print("-"*30)
            # print("Wathing for", model)
            print("Loading data from:", path)
            print("-"*30)
            dfs[model] = load_multilang_json(path)
            
            print(dfs[model].shape)

            # for language in dfs[model].language.unique():
            #     print(f"Language: {language}, Count: {dfs[model].language.value_counts()[language]}")

            print("Slovene Count:", dfs[model].language.value_counts()["SL"])
            print("Non-Slovene Count:", dfs[model].language.value_counts()["EN"])

            # show distribution of relative output length in comparison to the original text:
            dfs[model]["relative_length"] = dfs[model]['sl_translation'].str.len() / dfs[model]['text'].str.len()
            print("2 percentile: ", dfs[model]["relative_length"].quantile(0.02))
            print("5 percentile: ", dfs[model]["relative_length"].quantile(0.05))
            print("10 percentile: ", dfs[model]["relative_length"].quantile(0.1))
            print("90 percentile: ", dfs[model]["relative_length"].quantile(0.9))
            print("95 percentile: ", dfs[model]["relative_length"].quantile(0.95))
            print("98 percentile: ", dfs[model]["relative_length"].quantile(0.98))

        print("-"*30)
        print()
        
        ids_match = 0
        for i in range(len(dfs["1"])):
            if dfs["1"].iloc[i]["id"] in dfs["2"]["id"].values:
                ids_match += 1

        print("IDs match:", ids_match)

        df_join = dfs["1"].merge(dfs["2"], on="id", suffixes=("_1", "_2"), how="inner")
        df_join = df_join.drop(columns=["text_1", "title_1", "url_1", "Problematic_1", "Problematic_2"])
        df_join = df_join.rename(columns={"text_2": "text", "title_2": "title", "url_2": "url"})
        print("Fields of the joined dataframe:")
        for column in df_join.columns:
            print(f" - {column}")
        
        print("Shape of the joined dataframe:", df_join.shape)

        df_join.to_json(args.paired_data_path, orient="records", lines=True, force_ascii=False)
        print("Paired data saved to:", args.paired_data_path)

        # df_join.to_json(f"paired_data_with_scores{f'_{id}' if id>0 else ''}.jsonl", orient="records", lines=True, force_ascii=False)
        # print(f"Paired data saved to 'paired_data_with_scores{f'_{id}' if id>0 else ''}.jsonl'")
    