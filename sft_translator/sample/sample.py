#!/usr/bin/env python3
"""
Sample exactly N examples from a JSONL file and convert them to the target JSON format.

Input JSONL schema (per line):
  {
    "prompt":   str,
    "chosen":   str,
    "rejected": str,
    "src":      str
  }

Output JSON (list of length N):
  [
    {
      "id_session": "0",            # ... "199"
      "messages_a": [
        {"role": "user",      "content": <prompt>},
        {"role": "assistant", "content": <chosen>},
        {"role": "user",      "content": <src>}
      ],
      "messages_b": [
        {"role": "user",      "content": <prompt>},
        {"role": "assistant", "content": <rejected>},
        {"role": "user",      "content": <src>}
      ],
      "chosen": "A"                 # always "A"
    },
    ...
  ]
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_SEED = 123456789  # fixed seed for reproducibility
DEFAULT_N = 200


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON parse error at line {ln}: {e}") from e
            rows.append(obj)
    return rows


def validate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only rows with string fields: prompt, chosen, rejected, src."""
    out = []
    for r in rows:
        if (
            isinstance(r, dict)
            and isinstance(r.get("prompt"), str)
            and isinstance(r.get("chosen"), str)
            and isinstance(r.get("rejected"), str)
            and isinstance(r.get("src"), str)
        ):
            out.append(r)
    return out


def convert_sample(sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for idx, r in enumerate(sample):
        prompt = r["prompt"]
        chosen = r["chosen"]
        rejected = r["rejected"]
        src = r["src"]
        out.append(
            {
                "id_session": str(idx),
                "messages_a": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                    {"role": "user", "content": "Chosen: " + src},
                ],
                "messages_b": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected},
                    {"role": "user", "content": "Rejected: " + ("gams_9b_sft_translator" if src == "gams_27b" else "gams_27b") },
                ],
                "chosen": "A",
            }
        )
    return out


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Sample and convert JSONL to review JSON.")
    p.add_argument("input", help="Path to input .jsonl file")
    p.add_argument("output", help="Path to output .json file")
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of examples to sample (default: {DEFAULT_N})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    args = p.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")

    rows = read_jsonl(in_path)
    valid = validate_rows(rows)
    if len(valid) < args.n:
        raise SystemExit(
            f"Not enough valid examples: requested {args.n}, found {len(valid)} "
            f"(rows must have string fields: prompt, chosen, rejected, src)."
        )

    # Reproducible sampling
    rng = random.Random(args.seed)
    sample = rng.sample(valid, args.n)  # without replacement

    converted = convert_sample(sample)

    # Write pretty JSON with UTF-8 (preserve newlines and markdown)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(converted)} examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))