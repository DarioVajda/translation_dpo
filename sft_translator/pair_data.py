import argparse
import os
import json

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_path}: {e}")
                continue
    return data

def pair_data(first_input_path, first_text_field, second_input_path, second_text_field, output_path):
    first_data = load_data(first_input_path)
    second_data = load_data(second_input_path)
    print(f"Loaded {len(first_data)} examples from first file and {len(second_data)} from second file.")

    # print("-----")
    # print(first_data[-3]['text'][:50])
    # print("-----")
    # print(second_data[-3]['text'][:50])
    # print("-----")
    # return

    #---- conversation_id
    #---- text
    #---- role
    #---- Prompt
    #---- Problematic
    #---- gams_27b_translation
    #---- (gams_27b_translation_markdown_good)
    #---- (gams_27b_translation_markdown_judging)
    #---- sl_translation
    #---- sl_translation_markdown_good
    #---- sl_translation_markdown_judging

    paired_data = []
    for i in range(1, min(len(first_data), len(second_data)) + 1):
        if i % 1000 == 0:
            print(f"Paired {i}/{min(len(first_data), len(second_data))} examples...")
        
        first_example = first_data[-i]
        range_width = 1000
        from_range, to_range = -i-range_width, (-i+range_width if i > range_width else len(second_data))
        second_example = [ e for e in second_data[from_range:to_range] if e['text'] == first_example['text'] ]
        if len(second_example) == 0:
            print(f"{len(second_data[from_range:to_range])} No match for -{i}: {first_example['text'][:50]}")
            continue
            # print(f"Looking in range -{i-1000} to -{i+1000}")
            # print(second_data[-i]['text'][:50])
            # break
        if len(second_example) > 1:
            print(f"Multiple matches for: {first_example['text'][:50]}")
            continue
        second_example = second_example[0]

        paired_data.append({
            "text": first_example['text'],
            "conversation_id": first_example['conversation_id'],
            "role": first_example['role'],
            "Prompt": first_example['Prompt'],
            f"{first_text_field}": first_example[first_text_field],
            f"{first_text_field}_markdown_good": first_example[f"{first_text_field}_markdown_good"],
            f"{second_text_field}": second_example[second_text_field],
            f"{second_text_field}_markdown_good": second_example[f"{second_text_field}_markdown_good"],
        })
    print(f"Paired {len(paired_data)} examples.")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for example in paired_data:
            f_out.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"Saved paired data to {output_path}.")

def parse_args():
    # RUNNING THIS SCRIPT:
    # python3 pair_data.py --first_input_path=judged_2/data_sl_translation.jsonl --second_input_path=judged_2/data_gams_27b_translation.jsonl --output_path=judged_paired_2.jsonl
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first_input_path",
        type=str,
        default="judged/data_sl_translation.jsonl",
        help="Path to the first JSONL input file."
    )
    parser.add_argument(
        "--first_text_field",
        type=str, 
        default="sl_translation",
        help="Text field in the first input file that will be judged."
    )
    parser.add_argument(
        "--second_input_path",
        type=str,
        default="judged/data_gams_27b_translation.jsonl",
        help="Path to the second JSONL input file."
    )
    parser.add_argument(
        "--second_text_field",
        type=str, 
        default="gams_27b_translation",
        help="Text field in the second input file that will be judged."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="judged_paired.jsonl",
        help="Path to the output JSONL file."
    )
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    pair_data(
        args.first_input_path, 
        args.first_text_field,
        args.second_input_path, 
        args.second_text_field,
        args.output_path
    )