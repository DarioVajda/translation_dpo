from datasets import load_dataset
import json
import pandas as pd


def main():
    # Load the data
    data = pd.read_json("../language_identification/paired_data.jsonl", orient="records", lines=True)
    # Check the shape of the data
    print("Shape of the data:", data.shape)

    # I want to save the rows of data where exactly one of the columns language_eurollm or language_gams is "SL" and the other is something else
    gams_sl = data[(data["language_eurollm"] != "SL") & (data["language_gams"] == "SL")]
    eurollm_sl = data[(data["language_eurollm"] == "SL") & (data["language_gams"] != "SL")]
    both_bad = data[(data["language_eurollm"] != "SL") & (data["language_gams"] != "SL")]

    SIZE_THRESHOLD = 0.7

    # Preference dataset (gams)
    gams_sl = gams_sl.drop(columns=["id", "language_eurollm", "language_gams", "text", "title", "url", "Prompt_eurollm"])
    gams_sl = gams_sl.rename(columns={"Prompt_gams": "prompt", "sl_translation_gams": "chosen", "sl_translation_eurollm": "rejected"})
    gams_sl = gams_sl[gams_sl["relative_length_gams"] > SIZE_THRESHOLD]
    gams_sl = gams_sl.drop(columns=["relative_length_gams", "relative_length_eurollm"])
    gams_sl['src'] = "gams"

    # Preference dataset (eurollm)
    eurollm_sl = eurollm_sl.drop(columns=["id", "language_eurollm", "language_gams", "text", "title", "url", "Prompt_eurollm"])
    eurollm_sl = eurollm_sl.rename(columns={"Prompt_gams": "prompt", "sl_translation_eurollm": "chosen", "sl_translation_gams": "rejected"})
    eurollm_sl = eurollm_sl[eurollm_sl["relative_length_eurollm"] > SIZE_THRESHOLD]
    eurollm_sl = eurollm_sl.drop(columns=["relative_length_gams", "relative_length_eurollm"])
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

    # Save the data
    bad_lang_examples.to_json("bad_lang_examples.jsonl", orient="records", lines=True, force_ascii=False)

if __name__=="__main__":
    main()
