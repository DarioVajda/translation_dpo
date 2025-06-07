import os
import json
import pandas as pd

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
    

if __name__ == "__main__":
    # for id in [0, 1, 2]:
    for id in [1, 2]:
        dfs = {}
        print('*'*60)
        print(f"Processing data with id {id}")
        print('*'*60)

        # Load the data
        for model in [ "eurollm", "gams" ]:
            print("-"*30)
            print("Wathing for", model)
            print("-"*30)
            dfs[model] = load_multilang_json(f"./{model}9b_language_id_with_scores{f'_{id}' if id>0 else ''}")
            
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
        for i in range(len(dfs["eurollm"])):
            if dfs["eurollm"].iloc[i]["id"] in dfs["gams"]["id"].values:
                ids_match += 1

        print("IDs match:", ids_match)

        df_join = dfs["eurollm"].merge(dfs["gams"], on="id", suffixes=("_eurollm", "_gams"), how="inner")
        df_join = df_join.drop(columns=["text_eurollm", "title_eurollm", "url_eurollm", "Problematic_eurollm", "Problematic_gams"])
        df_join = df_join.rename(columns={"text_gams": "text", "title_gams": "title", "url_gams": "url"})
        print("Fields of the joined dataframe:")
        for column in df_join.columns:
            print(f" - {column}")
        
        print("Shape of the joined dataframe:", df_join.shape)

        df_join.to_json(f"paired_data_with_scores{f'_{id}' if id>0 else ''}.jsonl", orient="records", lines=True, force_ascii=False)

        print(f"Paired data saved to 'paired_data_with_scores{f'_{id}' if id>0 else ''}.jsonl'")
    