import json

def count_words_and_rows_in_text_fields(file_paths):
    total_word_count = 0
    total_row_count = 0
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Assuming the text field is named "text"
                if 'text' in data:
                    total_row_count += 1
                    total_word_count += len(data['text'].split())
    return total_word_count, total_row_count

# Example usage
file_paths = [
    "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_eurollm9b_translation.jsonl",
    "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_eurollm9b_translation_1.jsonl",
    "/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_eurollm9b_translation_2.jsonl"
]

total_word_count, total_row_count = count_words_and_rows_in_text_fields(file_paths)
print(f"Total word count in 'text' fields: {total_word_count}")
print(f"Total row count in 'text' fields: {total_row_count}")