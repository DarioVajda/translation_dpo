import sys
import json
import os
from datasets import Dataset

def load_data():
    # Load wiki dataset from path /ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/eval_datasets/wikipedia_eval.jsonl
    with open('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/eval_datasets/wikipedia_eval.jsonl', 'r') as f:
        wikipedia_dataset = [json.loads(line) for line in f]
    # Load ccnews dataset from path /ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/eval_datasets/ccnews_eval.jsonl
    with open('/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/eval_datasets/ccnews_eval.jsonl', 'r') as f:
        ccnews_dataset = [json.loads(line) for line in f]

    # Modify the keys in the ccnews_dataset
    for example in ccnews_dataset:
        example['text'] = example['plain_text']
        example['id'] = example['requested_url']
        example['url'] = example['requested_url']

    # Combine the datasets
    combined_dataset = wikipedia_dataset + ccnews_dataset

    # convert to a datasets Dataset object
    return Dataset.from_list(combined_dataset)
    

def selection(n, m, id):
    return [ i for i in range(n) ]
