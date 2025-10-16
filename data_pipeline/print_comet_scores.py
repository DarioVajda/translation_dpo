#!/usr/bin/env python3
import argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    sums, counts = {}, {}
    total_sum, total_count = 0.0, 0

    # read jsonl
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ds, score = obj.get("dataset"), obj.get("comet_score")
            if ds is None or score is None:
                continue
            score = float(score)
            sums[ds] = sums.get(ds, 0.0) + score
            counts[ds] = counts.get(ds, 0) + 1
            total_sum += score
            total_count += 1

    # prepare rows
    rows = [("Overall", total_count, total_sum / total_count if total_count else 0.0)]
    for ds in sorted(sums.keys()):
        rows.append((ds, counts[ds], sums[ds] / counts[ds]))

    # column widths
    col1 = max(len("Dataset"), max(len(r[0]) for r in rows))
    col2 = max(len("Count"), max(len(str(r[1])) for r in rows))
    col3 = len("Average COMET")

    # write table
    with open(args.output_path, "w", encoding="utf-8") as out:
        header = f"| {'Dataset'.ljust(col1)} | {'Count'.rjust(col2)} | {'Average COMET'.rjust(col3)} |"
        sep = "+" + "-"*(col1+2) + "+" + "-"*(col2+2) + "+" + "-"*(col3+2) + "+"
        out.write(sep + "\n" + header + "\n" + sep.replace("-", "=") + "\n")
        for name, cnt, avg in rows:
            out.write(f"| {name.ljust(col1)} | {str(cnt).rjust(col2)} | {avg:>{col3}.6f} |\n")
        out.write(sep + "\n")

if __name__ == "__main__":
    main()