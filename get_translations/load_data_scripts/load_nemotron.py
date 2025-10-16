import json, os
from datasets import Dataset


def split_conversation(conversation, i):
    return [
        { 
            'conversation_id': i,
            'text': og_msg['content'], 
            'gams_27b_translation': trans_msg['content'],
            'role': og_msg['role'], 
        } 
        for og_msg, trans_msg in zip(conversation['conversation_original']['messages'], conversation['conversation_translated']['messages'])
    ]

def load_file_data(file_path):
    json_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_path}: {e}")
                continue
    return json_list

def load_data():
    workspace_path = "/workspace" if os.path.exists("/workspace") else "/ceph/hpc/data/s24o01-42-users"
    path = workspace_path + "/data/nemotron_lmsys/data_out"
    # iterate through all jsonl files in the directory
    all_conversations = []
    for file in sorted(os.listdir(path)):
        file_data_conversations = load_file_data(os.path.join(path, file))
        all_conversations = all_conversations + file_data_conversations

    all_data = []
    for i, conv in enumerate(all_conversations):
        all_data = all_data + split_conversation(conv, i)

    return Dataset.from_list(all_data)


def selection(n, m, id):
    return [i for i in range(n) if i % (4) == id]
    # return [i for i in range(n) if i % (2) == id][:m] # use only 100 examples for testing

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} examples.")

    for i in range(10):
        print(data[i])
        print('---')

# Total messages: 192,206
# OLD: Valid id values: 0-19 (each batch has ~9600 messages)
# Valid id value: 2 (each batch has ~9600 messages)


# Interesting conversations:
# 86 - code (i think ???)