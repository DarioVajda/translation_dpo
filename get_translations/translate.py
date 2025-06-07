import os
from argparse import ArgumentParser
from tqdm import tqdm
import json
import random

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_from_disk

from task_adapter import get_task_adapter

def load_data(input_path):
    with open(input_path, "r") as f_in:
        data = json.load(f_in)

    return data

def fixed_selection(n, m, id, seed=42):
    # # Create a list [1, 2, ..., n]
    # numbers = list(range(1, n + 1))
    # # Set the seed so that the shuffle is reproducible
    # random.seed(seed)
    # # Shuffle the list in place
    # random.shuffle(numbers)
    # # Return the first m elements
    # return numbers[:m]

    return [ i for i in range(n) if i % 300 == id ]


def correct_examples(model_path, input_path, output_path, gpu_memory_util, tp_size, id):
    # data = load_from_disk(os.path.join(input_path, "train"))
    data = load_from_disk('/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_en/train')
    task_adapter = get_task_adapter(model_path)

    # Select first 100 examples
    # data = data.select(range(100))

    # Select random 20000 examples
    data = data.select(fixed_selection(len(data), 20000, id))
    # data = data.select(range(200))

    data_size = len(data)
    print("Number of examples:", data_size)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_util,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        seed=42
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=8192)

    print("Preparing prompts ...")
    def example_to_prompt(example):
        wikipedia_text = f"# {example['title']}\n\n{example['text']}"
        conversation = task_adapter.create_prompt(wikipedia_text)
        prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        problematic = len(tokenizer.encode(prompt)) > 4096

        return {"Prompt": prompt, "Problematic": problematic}
    
    prompt_data = data.map(example_to_prompt, num_proc=8)
    prompt_data = prompt_data.filter(lambda example: not example["Problematic"], num_proc=8)

    prompts = prompt_data["Prompt"]
    print("Number of prompts:", len(prompts))


    print("Running translations ...")
    responses = model.generate(prompts, sampling_params=sampling_params)

    def get_translation(example, idx):
        translation = responses[idx].outputs[0].text
        # print(translation)
        example["sl_translation"] = translation

        return example

    print("Processing translations ...")
    translation_data = prompt_data.map(get_translation, with_indices=True)

    # Save the data
    output_path = output_path[:-6] + f"_{id}.jsonl"
    print("Saving translations to", output_path)
    f_out =  open(output_path, "w")
    for example in tqdm(translation_data):
        write_example = example.copy()
        f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
    f_out.close()
    print("Done!")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (either HF ID or local path)."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input dataset."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output JSON file."
    )
    parser.add_argument(
        "--gpu_memory_util",
        type=float,
        required=True,
        help="GPU Memory utilization for vLLM graph. Float between 0 and 1."
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        required=True,
        help="Tensor parallel size of the model."
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="ID of the translation taks."
    )
    return parser.parse_args()


if __name__=="__main__":
    args=parse_args()
    correct_examples(args.model_path, args.input_path, args.output_path, args.gpu_memory_util, args.tp_size, args.id)
