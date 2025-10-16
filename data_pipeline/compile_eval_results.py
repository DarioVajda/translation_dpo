#!/usr/bin/env python3
"""
Summarize model evaluation results across folders into a Markdown table.

Adds bolding for the best value in each metric column:
- COMET columns (overall and per-dataset): highest value is best.
- Error-rate columns (Bad Lang, Short, Bad Markdown): lowest value is best.
- Ties are bolded for all tied cells. Missing values ("/") are never bolded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import math
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Hardcoded inputs (pairwise)
# ----------------------------
paths = [
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/bielik_11b_v2-6_instruct_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/eurollm_9b_instruct_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/gams_9b_dpo_translator_v0_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/gams_9b_instruct_dpo_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/gams_9b_instruct_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/gams_9b_sft_translator_dpo_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/gams_9b_sft_translator_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/gams_27b_instruct_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/gams_9b_sft_translator_dpo_full_eval",
    "/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_results/gemini-2.5-flash_eval",
]
paths = [ path+"/results" for path in paths]
models = [
    "Bielik-11B-v2.6-Instruct",
    "EuroLLM-9B-Instruct",
    "GaMS-9B-DPO-Translator-v0",
    "GaMS-9B-Instruct + DPO",
    "GaMS-9B-Instruct",
    "GaMS-9B-SFT-Translator + DPO",
    "GaMS-9B-SFT-Translator",
    "GaMS-27B-Instruct",
    "GaMS-9B-SFT-Translator-DPO-Full",
    "gemini-2.5-flash",
]

# ----------------------------
# Parsing helpers
# ----------------------------

def parse_comet_scoring(file_path: Path) -> Tuple[Optional[float], Dict[str, float]]:
    if not file_path.is_file():
        return None, {}
    overall = None
    per_dataset: Dict[str, float] = {}
    row_re = re.compile(r"^\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = row_re.match(line)
                if not m:
                    continue
                col1, col2, col3 = (c.strip() for c in m.groups())
                if col1.lower() in {"dataset", "metric"}:
                    continue
                try:
                    avg = float(col3)
                except ValueError:
                    continue
                name = col1.strip()
                if name.lower() == "overall":
                    overall = avg
                else:
                    per_dataset[name] = avg
    except Exception:
        return None, {}
    return overall, per_dataset


def parse_bad_language(file_path: Path) -> Optional[float]:
    if not file_path.is_file():
        return None
    pat = re.compile(r"Bad translation error rate:\s*([0-9]*\.?[0-9]+)\s*%?\b", re.IGNORECASE)
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            m = pat.search(content)
            if m:
                val = float(m.group(1))
                # if val <= 1.0:
                #     val *= 100.0
                return val
    except Exception:
        return None
    return None


def parse_short_error(file_path: Path) -> Optional[float]:
    if not file_path.is_file():
        return None
    prefer_pat = re.compile(
        r"Percentage of examples with translation too short:\s*([0-9]*\.?[0-9]+)\s*%", re.IGNORECASE
    )
    fallback_pat = re.compile(r"\(([0-9]*\.?[0-9]+)\s*%\)")
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            m = prefer_pat.search(content)
            if m:
                return float(m.group(1))
            m2 = fallback_pat.search(content)
            if m2:
                return float(m2.group(1))
    except Exception:
        return None
    return None


def parse_markdown_error(file_path: Path) -> Optional[float]:
    if not file_path.is_file():
        return None
    pat = re.compile(r"Error\s*rate.*?([0-9]*\.?[0-9]+)\s*%", re.IGNORECASE)
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    return float(m.group(1))
    except Exception:
        return None
    return None

# ----------------------------
# Formatting helpers
# ----------------------------

def fmt_comet(x: Optional[float]) -> str:
    return f"{x:.6f}" if isinstance(x, (int, float)) else "/"

def fmt_pct(x: Optional[float]) -> str:
    return f"{x:.2f}%" if isinstance(x, (int, float)) else "/"

def bold_if(s: str, make_bold: bool) -> str:
    return f"**{s}**" if make_bold and s != "/" else s

def is_close(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

# ----------------------------
# Core logic
# ----------------------------

def collect_results(paths: List[str], models: List[str]):
    assert len(paths) == len(models), "paths and models must be the same length"
    results = []
    dataset_union = set()
    for model, p in zip(models, paths):
        base = Path(p)
        comet_fp = base / "comet_scoring.txt"
        badlang_fp = base / "count_bad_lang.txt"
        short_fp = base / "count_short.txt"
        mdjudg_fp = base / "markdown_judging.txt"

        overall, per_ds = parse_comet_scoring(comet_fp)
        bad_lang = parse_bad_language(badlang_fp)
        short_err = parse_short_error(short_fp)
        md_err = parse_markdown_error(mdjudg_fp)

        dataset_union.update(per_ds.keys())

        results.append(
            {
                "model_name": model,
                "overall_comet": overall,
                "datasets": per_ds,
                "bad_lang_pct": bad_lang,
                "short_pct": short_err,
                "md_pct": md_err,
            }
        )
    all_datasets = sorted(dataset_union)
    return results, all_datasets


def sort_results(results: List[dict]) -> List[dict]:
    def key_fn(item: dict):
        ov = item.get("overall_comet")
        return (ov is None, 0 if ov is None else -ov)
    return sorted(results, key=key_fn)

def compute_bests(results: List[dict], datasets: List[str]):
    """
    Determine the best values for bolding:
      - COMET (overall & per-dataset): max
      - Error rates: min
    Returns a dict with keys:
      'overall', 'per_ds' (dict), 'bad_lang', 'short', 'md'
    """
    # overall comet: max among non-None
    overall_vals = [r["overall_comet"] for r in results if r.get("overall_comet") is not None]
    overall_best = max(overall_vals) if overall_vals else None

    # per-dataset comet: max for each dataset
    per_ds_best: Dict[str, Optional[float]] = {}
    for ds in datasets:
        vals = []
        for r in results:
            val = r.get("datasets", {}).get(ds)
            if val is not None:
                vals.append(val)
        per_ds_best[ds] = max(vals) if vals else None

    # error rates: min among non-None
    def min_or_none(key: str) -> Optional[float]:
        vals = [r[key] for r in results if r.get(key) is not None]
        return min(vals) if vals else None

    return {
        "overall": overall_best,
        "per_ds": per_ds_best,
        "bad_lang": min_or_none("bad_lang_pct"),
        "short": min_or_none("short_pct"),
        "md": min_or_none("md_pct"),
    }

def render_markdown(results: List[dict], datasets: List[str], bests: dict) -> str:
    headers = ["Model", "Overall Comet"] + datasets + ["Bad Lang (%)", "Short (%)", "Bad Markdown (%)"]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for r in results:
        row: List[str] = []
        row.append(r["model_name"])

        # overall comet
        ov = r.get("overall_comet")
        row.append(bold_if(fmt_comet(ov), is_close(ov, bests["overall"])))

        # per-dataset comet
        per_ds = r.get("datasets", {})
        for ds in datasets:
            val = per_ds.get(ds)
            row.append(bold_if(fmt_comet(val), is_close(val, bests["per_ds"].get(ds))))

        # error rates (lower is better)
        bl = r.get("bad_lang_pct")
        sh = r.get("short_pct")
        md = r.get("md_pct")

        row.append(bold_if(fmt_pct(bl), is_close(bl, bests["bad_lang"])))
        row.append(bold_if(fmt_pct(sh), is_close(sh, bests["short"])))
        row.append(bold_if(fmt_pct(md), is_close(md, bests["md"])))

        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines) + "\n"

def resolve_output_path(arg: str) -> Path:
    out = Path(arg)
    if out.suffix.lower() == ".md":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    else:
        out.mkdir(parents=True, exist_ok=True)
        return out / "summary.md"

def main():
    parser = argparse.ArgumentParser(description="Summarize model results into a Markdown table.")
    parser.add_argument(
        "output",
        help="Output destination. If a .md path is given, write directly to it; otherwise treat as a directory and write summary.md inside."
    )
    args = parser.parse_args()

    out_path = resolve_output_path(args.output)

    results, datasets = collect_results(paths, models)
    results_sorted = sort_results(results)
    bests = compute_bests(results, datasets)  # compute on all results (order-independent)
    md = render_markdown(results_sorted, datasets, bests)

    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()