import os
import json
import pandas as pd


def load_multilang_json(base_dir, subfile):
    """
    Load all JSONL files from subfolders of base_dir.
    Each JSON object gets a 'lang' field based on its subfolder.
    
    Args:
        base_dir (str): Path to the root directory (e.g., "eurollm_translations/")
    
    Returns:
        pd.DataFrame: Combined DataFrame with all entries and added 'lang' field.
    """
    all_data = []

    # Loop through each subdirectory
    for lang_code in os.listdir(base_dir):
        lang_dir = os.path.join(base_dir, lang_code)
        if not os.path.isdir(lang_dir):
            continue
        
        file_path = os.path.join(lang_dir, subfile)
        if not os.path.isfile(file_path):
            print(f"Skipping: {file_path} (not found)")
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    obj["lang"] = lang_code
                    all_data.append(obj)
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {file_path}")

    return pd.DataFrame(all_data)

folders = [ 
    "/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/language_id/eurollm_translations",
    "/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/language_id/gams_translations",
    "/ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/language_id/gams_dpo_translations"
]
subfiles = [
    "eurollm_translations.jsonl",
    "gams_translations.jsonl",
    "gams_dpo_translations.jsonl"
]

for folder, subfile in zip(folders, subfiles):
    print(f"Processing folder: {folder}")
    # Load the JSON files from the folder
    df = load_multilang_json(folder, subfile)
    
    # Save the DataFrame to a single JSONL file
    output_file = f"{folder}.jsonl"
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    
    print(f"Saved merged translations to: {output_file}")