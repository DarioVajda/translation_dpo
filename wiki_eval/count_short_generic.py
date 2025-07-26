import argparse
import json
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Count translations that are too short or have bad language."
    )
    parser.add_argument(
        '--sl_translations_file',
        type=str,
        required=True,
        help='Path to the Slovene translations JSONL file.'
    )
    args = parser.parse_args()
    input_path = args.sl_translations_file

    # open ./language_id/gams_dpo_translations.jsonl
    translation_list = []
    with open(input_path, "r") as file:
        for line in file:
            translation_list.append(json.loads(line.strip()))


    too_short = 0
    total = 0
    for example in translation_list:
        if example["lang"] != "SL":
            continue
        total += 1
        # print(example["sl_translation"])
        # print(example["text"])
        # print('-' * 60)
        if len(example["sl_translation"]) / len(example["text"]) < 0.7:
            too_short += 1
    print(f"Number of examples with translation too short: {too_short} out of {len(translation_list)} ({too_short / len(translation_list) * 100:.2f}%)")
    print(f"Percentage of examples with translation too short: {too_short / total * 100:.2f}%")


if __name__ == "__main__":
    main()