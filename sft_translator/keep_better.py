#!/usr/bin/env python3
import argparse
import json
import sys
from typing import List, Tuple, Optional

# Use a non-interactive backend so this works on headless machines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def is_thematic_break_line(line: str) -> bool:
    """
    Detect a Markdown thematic break line:
    - three or more of the same char among '-', '*', '_'
    - spaces allowed between the characters
    Examples that should match: '---', '***', '___', '- - -', '* * *', '_ _ _', '------'
    """
    s = line.strip()
    # Remove internal spaces to support '- - -', '* * *', etc.
    s_no_space = s.replace(" ", "")
    if len(s_no_space) < 3:
        return False
    if set(s_no_space) <= {"-"}:
        return True
    if set(s_no_space) <= {"*"}:
        return True
    if set(s_no_space) <= {"_"}:
        return True
    return False


def is_markdown_heavy(text: str) -> bool:
    """Return True if the text contains at least MIN_THEMATIC_BREAKS thematic break lines."""
    MIN_THEMATIC_BREAKS = 1
    count = 0
    for line in text.splitlines():
        if is_thematic_break_line(line):
            count += 1
            if count >= MIN_THEMATIC_BREAKS:
                return True
    return False


def read_lengths_by_group(
    path: str,
    filtered_output_path: Optional[str] = None,
    min_len: int = 2000,
) -> Tuple[List[int], List[int]]:
    """
    Returns (heavy_lengths, other_lengths) lists of character counts for 'prompt'.
    Optionally writes filtered entries to JSONL if filtered_output_path is provided.
    """
    heavy_lengths: List[int] = []
    other_lengths: List[int] = []
    bad_json = 0
    missing_prompt = 0
    saved_count = 0

    out_f = open(filtered_output_path, "w", encoding="utf-8") if filtered_output_path else None

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                raw = line.rstrip("\n")
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue

                if "prompt" not in obj or obj["prompt"] is None:
                    missing_prompt += 1
                    continue

                prompt_text = str(obj["prompt"])
                length = len(prompt_text)
                heavy = is_markdown_heavy(prompt_text)

                if heavy:
                    heavy_lengths.append(length)
                else:
                    other_lengths.append(length)

                # Save filtered examples if requested
                if out_f is not None:
                    if (length > min_len) or heavy:
                        # Write normalized JSON to the output file
                        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        saved_count += 1
    finally:
        if out_f is not None:
            out_f.close()

    if bad_json or missing_prompt:
        print(
            f"Note: skipped {bad_json} bad JSON lines and {missing_prompt} entries without a usable 'prompt'.",
            file=sys.stderr,
        )
    if filtered_output_path:
        print(f"Wrote {saved_count} filtered examples to {filtered_output_path}.", file=sys.stderr)

    if not heavy_lengths and not other_lengths:
        print("No valid 'prompt' lengths were found. Exiting.", file=sys.stderr)
        sys.exit(1)

    return heavy_lengths, other_lengths


def plot_stacked_histogram(
    heavy_lengths: List[int],
    other_lengths: List[int],
    out_path: str = "len_distribution.png",
) -> None:
    """
    Make a stacked histogram with blue for markdown-heavy (bottom) and red for others (top).
    """
    # Determine shared bin edges using all data
    all_lengths = np.array(heavy_lengths + other_lengths, dtype=np.int32)
    counts_all, bin_edges = np.histogram(all_lengths, bins="auto")

    # Histogram per group using the same bins
    heavy_counts, _ = np.histogram(np.array(heavy_lengths, dtype=np.int32), bins=bin_edges)
    other_counts, _ = np.histogram(np.array(other_lengths, dtype=np.int32), bins=bin_edges)

    # Bar positions and widths
    lefts = bin_edges[:-1]
    widths = np.diff(bin_edges)

    plt.figure(figsize=(10, 6))

    # Bottom: markdown-heavy (blue) — note: current rule is ≥1 thematic break
    plt.bar(
        lefts,
        heavy_counts,
        width=widths,
        align="edge",
        label="Markdown-heavy (≥1 thematic break)",
        edgecolor="black",
        linewidth=0.4,
        color="blue",
    )
    # Top: other (red), stacked on heavy
    plt.bar(
        lefts,
        other_counts,
        width=widths,
        align="edge",
        bottom=heavy_counts,
        label="Other",
        edgecolor="black",
        linewidth=0.4,
        color="red",
    )

    plt.title("Stacked Distribution of 'prompt' Character Lengths")
    plt.xlabel("Characters")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    # Some summary stats
    def safe_stats(values: List[int]):
        if not values:
            return (0, 0.0, 0.0, 0.0)
        arr = np.array(values, dtype=np.int32)
        return (len(arr), float(np.mean(arr)), float(np.median(arr)), float(np.percentile(arr, 95)))

    n_heavy, mean_heavy, med_heavy, p95_heavy = safe_stats(heavy_lengths)
    n_other, mean_other, med_other, p95_other = safe_stats(other_lengths)
    n_total = n_heavy + n_other
    mean_all = float(np.mean(all_lengths))
    med_all = float(np.median(all_lengths))
    p95_all = float(np.percentile(all_lengths, 95))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(
        f"Saved stacked histogram to {out_path}\n"
        f"TOTAL: {n_total} | Mean: {mean_all:.1f} | Median: {med_all:.1f} | 95th: {p95_all:.1f}\n"
        f"Markdown-heavy: n={n_heavy}, mean={mean_heavy:.1f}, median={med_heavy:.1f}, 95th={p95_heavy:.1f}\n"
        f"Other:          n={n_other}, mean={mean_other:.1f}, median={med_other:.1f}, 95th={p95_other:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a stacked histogram of character lengths of 'prompt' fields in a JSONL file.\n"
            "Blue = prompts with ≥1 Markdown thematic break (---/***/___). Red = the rest.\n"
            "Optionally save filtered examples to JSONL."
        )
    )
    parser.add_argument("jsonl_path", help="Path to the input .jsonl file")
    parser.add_argument(
        "--plot_path",
        default="len_distribution.png",
        help="Path to save the output plot (default: len_distribution.png)",
    )
    parser.add_argument(
        "--filtered_output_path",
        default=None,
        help="If set, write JSONL of entries where len(prompt) > min_len OR is_markdown_heavy(prompt).",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=2000,
        help="Length threshold for filtering (default: 2000).",
    )
    args = parser.parse_args()

    heavy_lengths, other_lengths = read_lengths_by_group(
        args.jsonl_path,
        filtered_output_path=args.filtered_output_path,
        min_len=args.min_len,
    )
    plot_stacked_histogram(heavy_lengths, other_lengths, args.plot_path)


if __name__ == "__main__":
    main()