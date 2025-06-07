import json
import argparse

from comet import download_model, load_from_checkpoint


def load_data(input_path):
    data = []
    
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            data.append(json.loads(line))

    return data


def save_data(output_path, processed_data):
    with open(output_path, "w", encoding="utf-8") as outfile:
        for item in processed_data:
            json.dump(item, outfile)
            outfile.write("\n")


def load_model():
    # model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")

    print("Model path:", model_path)
    model = load_from_checkpoint(model_path)

    return model


def score_dataset(input_path, output_path, batch_size):
    data = load_data(input_path)
    input_data = [{"src": f"{example['title']}\n\n{example['text']}", "mt": example["sl_translation"]} for example in data]

    model = load_model()
    model_output = model.predict(input_data, batch_size=batch_size, gpus=1)
    print("Dataset score:", model_output.system_score)

    scores = model_output.scores
    assert len(scores) == len(data), "Different length of COMET scores and input data"
    for example, score in zip(data, scores):
        example["comet_score"] = score

    save_data(output_path, data)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for COMET model from JSONL file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for the Comet prediction model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score_dataset(args.input_path, args.output_path, args.batch_size)
