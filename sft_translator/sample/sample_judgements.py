#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Output file format (JSON)
#
# The program writes a JSON array where each element is a dictionary:
#
# {
#   "id_session": "0",             # string, unique index for this record
#   "messages_a": [                # list of message turns in conversation
#       {
#         "role": "user",
#         "content": "original text from input field 'text'"
#       },
#       {
#         "role": "assistant",
#         "content": "translation from field '<translation_field>'"
#       },
#       {
#         "role": "user",
#         "content": "explanation from field '<translation_field>_markdown_judging'"
#       }
#   ],
#   "chosen": "A"                  # "A" if '<translation_field>_markdown_good' == "YES",
#                                  # otherwise "B"
# }
#
# Notes:
# - The file is always valid JSON, UTF-8 encoded.
# - "id_session" is sequential string IDs ("0", "1", ...).
# - Sampling is without replacement; combined list is shuffled using --seed.
# - Only required fields are carried forward; others are ignored.
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample from two JSONL files and format for judgement runs."
    )
    parser.add_argument(
        "--input_file",
        default="judged_2_gams/data_sl_translation.jsonl",
        help="Path to first input .jsonl file.",
    )
    parser.add_argument(
        "--input_file_b",
        default="judged_2_gams/data_gams_27b_translation.jsonl",
        help="Path to second input .jsonl file.",
    )
    parser.add_argument(
        "--output_file",
        default="judgement_sample_gams.json",
        help="Path to write the merged output JSON file.",
    )
    parser.add_argument(
        "--translation_field",
        default="sl_translation",
        help="Translation field name used for the first input file.",
    )
    parser.add_argument(
        "--translation_field_b",
        default="gams_27b_translation",
        help="Translation field name used for the second input file.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=300,
        help="Total number of examples to sample across BOTH files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible sampling.",
    )
    return parser.parse_args()

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[warn] Skipping malformed JSON at {path}:{lineno}: {e}\n")
    return items

def sample_and_transform(path, trans_field, n, rng):
    input_path = Path(path)
    if not input_path.exists():
        sys.stderr.write(f"[error] Input file not found: {input_path}\n")
        sys.exit(1)

    data = read_jsonl(input_path)
    if not data:
        sys.stderr.write(f"[error] No valid records found in input file: {input_path}\n")
        sys.exit(1)

    good_key = f"{trans_field}_markdown_good"
    judge_key = f"{trans_field}_markdown_judging"
    required = {"text", trans_field, good_key, judge_key}

    filtered = [obj for obj in data if required.issubset(obj.keys())]
    if not filtered:
        sys.stderr.write(
            f"[error] None of the records in {input_path} contain required fields: "
            f"{', '.join(sorted(required))}\n"
        )
        sys.exit(1)

    k = min(n, len(filtered))
    if k < n:
        sys.stderr.write(
            f"[warn] Requested {n} samples from {input_path} but only {len(filtered)} eligible. Using {k}.\n"
        )

    sampled = rng.sample(filtered, k=k)
    out = []
    for obj in sampled:
        chosen = "A" if str(obj.get(good_key, "")).strip().upper() == "YES" else "B"
        out.append({
            "id_session": "0",  # temporary; will be reassigned after merge+shuffle
            "messages_a": [
                {"role": "user", "content": obj["text"]},
                {"role": "assistant", "content": obj[trans_field]},
                {"role": "user", "content": obj[judge_key]},
            ],
            "chosen": chosen,
        })
    return out, k, len(filtered)

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Split sample size across A and B (remainder to B so total matches sample_size)
    n_a = args.sample_size // 2
    n_b = args.sample_size - n_a

    # Sample from each file independently with the same seeded RNG
    out_a, used_a, avail_a = sample_and_transform(args.input_file, args.translation_field, n_a, rng)
    out_b, used_b, avail_b = sample_and_transform(args.input_file_b, args.translation_field_b, n_b, rng)

    combined = out_a + out_b

    # Shuffle combined list for interleaving and fairness
    rng.shuffle(combined)

    # Reassign sequential id_session
    for i, item in enumerate(combined):
        item["id_session"] = str(i)

    # Final sanity/warning if overall shortfall
    total_used = used_a + used_b
    if total_used < args.sample_size:
        sys.stderr.write(
            f"[warn] Requested total sample_size={args.sample_size} but only "
            f"{total_used} eligible examples were available across files "
            f"({avail_a} in A, {avail_b} in B).\n"
        )

    # Write JSON
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(combined)} records to {out_path} "
        f"(A: requested {n_a}, used {used_a} / {avail_a}; "
        f"B: requested {n_b}, used {used_b} / {avail_b})"
    )

if __name__ == "__main__":
    main()