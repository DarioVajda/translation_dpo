import os, json, argparse


def get_translation_prompt(text):
    return f"Prevedi naslednje besedilo v slovenščino.\n{text}"

def fix_data_fields(example):
    # rename 'Prompt' to 'markdown_prompt' and then add field 'Prompt' with value get_translation_prompt(example['text'])
    fixed_example = example.copy()
    fixed_example['markdown_prompt'] = fixed_example['Prompt']
    fixed_example['Prompt'] = get_translation_prompt(fixed_example['text'])
    return fixed_example

def parse_args():
    # RUNNING THIS SCRIPT:
    # python3 create_dataset.py --input_file=judged_paired_2.jsonl --output_file=markdown_preference_dataset_2.jsonl

    # read --input_file, --output_file, --first_text_field, --second_text_field
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="judged_paired_2_train.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="markdown_final_train_dataset.jsonl",
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--first_text_field",
        type=str,
        default="sl_translation",
        help="Text field in the first input file that will be judged."
    )
    parser.add_argument(
        "--first_src",
        type=str,
        default="gams_9b_sft_translator",
        help="Model used for the first translation."
    )
    parser.add_argument(
        "--second_text_field",
        type=str,
        default="gams_27b_translation",
        help="Text field in the second input file that will be judged."
    )
    parser.add_argument(
        "--second_src",
        type=str,
        default="gams_27b",
        help="Model used for the second translation."
    )
    return parser.parse_args()

def filter_data(input_file, first_text_field, second_text_field, first_src, second_src, output_file):
    print(f"Loading data from {input_file}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data.append(fix_data_fields(json.loads(line)))
    print(f"Loaded {len(data)} examples.")

    print("Fields in the data:", data[0].keys())

    first_better = []
    second_better = []

    for example in data:
        first_good = example[f"{first_text_field}_markdown_good"]
        second_good = example[f"{second_text_field}_markdown_good"]

        if first_good == "YES" and second_good != "YES":
            first_better.append({
                "prompt": example["Prompt"],
                "chosen": example[first_text_field],
                "rejected": example[second_text_field],
                "src": first_src
            })
        elif first_good != "YES" and second_good == "YES":
            second_better.append({
                "prompt": example["Prompt"],
                "chosen": example[second_text_field],
                "rejected": example[first_text_field],
                "src": second_src
            })

    print(f"First ({first_src}) better: {len(first_better)} examples.")
    print(f"Second ({second_src}) better: {len(second_better)} examples.")

    filtered_data = first_better + second_better

    # save the data to output_file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for example in filtered_data:
            f_out.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Saved filtered data to {output_file}.")


def main():
    args = parse_args()
    filter_data(
        args.input_file,
        args.first_text_field,
        args.second_text_field,
        args.first_src,
        args.second_src,
        args.output_file,
    )


if __name__=="__main__":
    main()