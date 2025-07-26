import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser(
        description="Merge multilingual JSONL files from a single translations folder."
    )
    parser.add_argument(
        '--multilang_file',
        help='Path to the root translations directory.'
    )
    args = parser.parse_args()
    input_path = args.multilang_file

    translation_list = []
    if not os.path.exists(input_path):
        print(f"Input file {input_path} does not exist.")
        return
    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                translation = json.loads(line.strip())
                translation_list.append(translation)
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()}")
    
    if not translation_list:
        print("No valid translations found.")
        return
    
    count_slovene = len([t for t in translation_list if t.get('lang') == 'SL'])
    count_all = len(translation_list)

    print("Total translations:", count_all)
    print("Slovene translations:", count_slovene)
    print("Percentage of Slovene translations:", (count_slovene / count_all) * 100 if count_all > 0 else 0)
    print("Bad translation error rate:", (count_all - count_slovene) / count_all * 100 if count_all > 0 else 0)


if __name__ == "__main__":
    main()