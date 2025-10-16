#!/usr/bin/env python3
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Intersect two JSONL datasets, keeping only identical entries."
    )
    parser.add_argument(
        "--input1",
        default="markdown_preference_dataset_2.jsonl",
        help="Path to first input JSONL file.",
    )
    parser.add_argument(
        "--input2",
        default="markdown_preference_dataset_2_gams.jsonl",
        help="Path to second input JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="markdown_dataset.jsonl",
        help="Path to write the intersection JSONL file.",
    )
    return parser.parse_args()

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    args = parse_args()

    # Load both datasets
    data1 = read_jsonl(args.input1)
    data2 = read_jsonl(args.input2)

    # Convert to sets of JSON strings for exact matching
    set1 = {json.dumps(obj, sort_keys=True) for obj in data1}
    set2 = {json.dumps(obj, sort_keys=True) for obj in data2}

    # Intersection
    intersection = set1 & set2
    output_data = [json.loads(s) for s in intersection]

    # Write output
    write_jsonl(args.output, output_data)

    # Print stats
    print(f"Entries in {args.input1}: {len(data1)}")
    print(f"Entries in {args.input2}: {len(data2)}")
    print(f"Entries in intersection (written to {args.output}): {len(output_data)}")

if __name__ == "__main__":
    main()