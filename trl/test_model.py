print("Starting the script...")
from transformers import pipeline
print("imported pipeline from transformers")
import torch
print("imported torch")
from tqdm import tqdm
print("imported tqdm")

# model_id = "cjvt/GaMS-9B-Instruct"
# model_id = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct-translate-v2"
model_id = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct-translate-v3"

pline = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto", # replace with "mps" to run on a Mac device
)
print("Initialized pipeline with model:", model_id)

# message = [{
#     "role": "user",
#     "content": 
#         "Prevedi naslednje angleško besedilo v slovenščino.\n" +
#         "Today is a really nice day. The sun is shining and the birds are singing. I went for a walk in the park and saw many people enjoying the weather."
# }]

# print(message)

# response = pline(message, max_new_tokens=2048)

# print("Model's response:", response[0]["generated_text"][-1]["content"])
# translation_list.append(response[0]["generated_text"][-1]["content"])


# Load the dataset and prepare messages for translation
def get_messages():
    ensl_dataset_path = "/ceph/hpc/data/s24o01-42-users/slobench/data/test_data/translation/slobench_ensl.en.txt"
    with open(ensl_dataset_path, "r") as file:
        ensl_dataset = file.readlines()
    ensl_dataset = [line.strip() for line in ensl_dataset]

    messages = [ [{ "role": "user", "content": ("Prevedi naslednje angleško besedilo v slovenščino.\n"+text)}] for text in ensl_dataset ]
    return messages

messages = get_messages()

translation_list = []

# Iterate over the messages and generate translations
for message in tqdm(messages, desc="Translating"):
    response = pline(message, max_new_tokens=2048)
    # print("Model's response:", response[0]["generated_text"][-1]["content"])
    
    # Keep only the first line of the response
    res = response[0]["generated_text"][-1]["content"].split("\n")[0]
    translation_list.append(res)

# Save the translations to a file
output_file_path = "/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/slobench_ensl_translations3.txt"
with open(output_file_path, "w") as output_file:
    for translation in translation_list:
        output_file.write(translation + "\n")

print(f"Translations saved to {output_file_path}")