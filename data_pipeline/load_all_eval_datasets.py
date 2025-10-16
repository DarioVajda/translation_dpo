import sys
import json
import os
from datasets import Dataset

def load_data():
    all_datasets = {}
    root = '/ceph/hpc/data/s24o01-42-users/translation_optimization'
    # root = '/shared/workspace/povejmo/translation_optimization'
    data_paths, dataset_names = [
        f'{root}/data_pipeline/eval_datasets/wikipedia_eval.jsonl',
        f'{root}/data_pipeline/eval_datasets/ccnews_eval.jsonl',
        f'{root}/data_pipeline/eval_datasets/nemotron_eval.jsonl'
    ], [ "wikipedia", "ccnews", "nemotron" ]

    data_per_dataset = 500

    for path, name in zip(data_paths, dataset_names):
        with open(path, 'r') as f:
            all_datasets[name] = [{**json.loads(line), "dataset": name} for line in f][:data_per_dataset]

    # Modify the keys in the ccnews_dataset
    for example in all_datasets["ccnews"]:
        example['text'] = example['plain_text']
        example['id'] = example['requested_url']
        example['url'] = example['requested_url']

    # Modify the keys in the nemotron dataset
    for example in all_datasets["nemotron"]:
        example['id'] = "nemotron_" + str(example['conversation_id'])
        example['url'] = "nemotron_" + str(example['conversation_id'])

    # Combine the datasets
    combined_dataset = all_datasets["wikipedia"] + all_datasets["ccnews"] + all_datasets["nemotron"]

    # convert to a datasets Dataset object
    return Dataset.from_list(combined_dataset)
    

def selection(n, m, id):
    return [ i for i in range(n) ]
