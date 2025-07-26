import os
from argparse import ArgumentParser
from tqdm import tqdm
import json
import random
import importlib.util

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from task_adapter import get_task_adapter

def correct_examples(model_path, input_module, output_path, gpu_memory_util, tp_size, id):
    # loading the data with the provided input module
    print("Loading data ...")
    data = input_module.load_data()

    # selecting a fixed number of examples based on the id
    print("Selecting examples ...")
    data = data.select(input_module.selection(len(data), 100, id))

    data_size = len(data)
    print("Number of examples:", data_size)

    task_adapter = get_task_adapter(model_path)

    # check if the model path is a local path or a Hugging Face ID
    if os.path.isdir(model_path):
        print("Using local model path:", model_path)
    else:
        print("Using Hugging Face model ID:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LLM(
        model=model_path,
        tokenizer=model_path,
        gpu_memory_utilization=gpu_memory_util,
        trust_remote_code=True,
        enable_prefix_caching=False,
        tensor_parallel_size=tp_size,
        seed=42
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=8192)

    print("Preparing prompts ...")
    def example_to_prompt(example):
        text_field = "plain_text" if "plain_text" in example else "text"
        wikipedia_text = f"# {example['title']}\n\n{example[text_field]}"
        conversation = task_adapter.create_prompt(wikipedia_text)
        prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        problematic = len(tokenizer.encode(prompt)) > 4096

        return {"Prompt": prompt, "Problematic": problematic}

    prompt_data = data.map(
        example_to_prompt, 
        num_proc=8, 
        load_from_cache_file=False,
        keep_in_memory=True
    )
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
        "--input_module",
        type=str,
        required=True,
        help="Script for loading the data (should define a `load_data` function)."
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

def import_module_from_path(path: str):
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__=="__main__":
    args=parse_args()

    input_module = import_module_from_path(args.input_module)

    correct_examples(args.model_path, input_module, args.output_path, args.gpu_memory_util, args.tp_size, args.id)
