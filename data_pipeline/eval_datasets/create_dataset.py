from datasets import load_from_disk
import json
import os

def load_data(year):
    path = f"/shared/workspace/povejmo/corpuses/cc_news/ccnews_{year}_en.hf"
    if os.path.isdir(path):
        return load_from_disk(path)
    else:
        # print(f"⚠️  Skipping missing {path}")
        assert False, f"Dataset for year {year} not found at '{path}'"

def selection(n, m, id):
    return [i for i in range(n) if i % (300) == id][:m]

data = load_data(2019)
id = 299
eval_dataset_size = 500

eval_dataset = selection(len(data), eval_dataset_size, id)

# save dataset to /shared/workspace/povejmo/translation_optimization/data_pipeline/eval_datasets/ccnews_eval.jsonl
output_path = '/shared/workspace/povejmo/translation_optimization/data_pipeline/eval_datasets/ccnews_eval.jsonl'
with open(output_path, 'w') as f_out:
    for idx in eval_dataset:
        example = data[idx]
        f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Saved {len(eval_dataset)} examples to {output_path}")
    print("Done!")
