import os
import json
import pandas as pd
import argparse


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
        # print(lang_dir)
        if not os.path.isdir(lang_dir):
            continue

        # Check which files are in the language directory
        subfile = [f for f in os.listdir(lang_dir) if f.endswith(".jsonl")][0]
        # print(f"Processing file: {subfile} in {lang_dir}")
        
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
def main():
    parser = argparse.ArgumentParser(
        description="Merge multilingual JSONL files from a single translations folder."
    )
    parser.add_argument(
        "--multilang_dir",
        help="Path to the root translations directory.",
    )
    args = parser.parse_args()

    base_dir = args.multilang_dir
    subfile_name = base_dir + ".jsonl"  # Assuming the subfile is named after the base directory

    print(f"Processing folder: {base_dir}")
    print(f"Looking for subfile: {subfile_name}")

    df = load_multilang_json(base_dir, subfile_name)

    # Save merged output alongside the folder
    output_file = f"{base_dir}.jsonl"
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)

    print(f"Saved merged translations to: {output_file}")


if __name__ == "__main__":
    main()