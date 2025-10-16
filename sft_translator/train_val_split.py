import argparse, json, os, random, re

#region helper functions detecting markdown text
def is_thematic_break_line(line: str) -> bool:
    """
    Detect a Markdown thematic break line:
    - three or more of the same char among '-', '*', '_'
    - spaces allowed between the characters
    Examples that should match: '---', '***', '___', '- - -', '* * *', '_ _ _', '------'
    """
    s = line.strip()
    # Remove internal spaces to support '- - -', '* * *', etc.
    s_no_space = s.replace(" ", "")
    if len(s_no_space) < 3:
        return False
    if set(s_no_space) <= {"-"}:
        return True
    if set(s_no_space) <= {"*"}:
        return True
    if set(s_no_space) <= {"_"}:
        return True
    return False

def is_markdown_heavy(text: str) -> bool:
    """Return True if the text contains at least MIN_THEMATIC_BREAKS thematic break lines."""
    MIN_THEMATIC_BREAKS = 1
    count = 0
    for line in text.splitlines():
        if is_thematic_break_line(line):
            count += 1
            if count >= MIN_THEMATIC_BREAKS:
                return True
    return False

# High-signal Markdown constructs (GitHub/CommonMark-ish).
_MD_REGEXES = [
    re.compile(r'(?m)^(#{1,6})\s+\S'),                     # ATX headings:  # H1  .. ###### H6
    re.compile(r'(?m)^[^\n]+\n=+\s*$|^[^\n]+\n-+\s*$', re.M),  # Setext headings
    re.compile(r'!\[[^\]]*\]\([^)]+\)'),                   # Images: ![alt](url)
    re.compile(r'\[[^\]]+\]\([^)]+\)'),                    # Links:  [text](url)
    re.compile(r'(?m)^```'),                               # Fenced code block start
    re.compile(r'`[^`\n]+`'),                              # Inline code
    re.compile(r'(?m)^\s*([-*+])\s+\S'),                   # Unordered list item
    re.compile(r'(?m)^\s*\d{1,3}[.)]\s+\S'),               # Ordered list item
    re.compile(r'(?m)^\s*>+\s+\S'),                        # Blockquote
    re.compile(r'(?m)^\s*(-{3,}|\*{3,}|_{3,})\s*$'),       # Horizontal rule
    # Bold / emphasis (kept conservative to reduce false positives)
    re.compile(r'(\*\*|__)(?=\S).+?(?<=\S)\1'),            # **bold** or __bold__
    re.compile(r'(?<!\*)\*(?=\S)[^*\n]+(?<=\S)\*(?!\*)|(?<!_)_(?=\S)[^_\n]+(?<=\S)_(?!_)'),
    # Simple table hint: a '|' row followed by a dashed header separator
    re.compile(r'(?m)^\s*\|.+\|\s*$\n^\s*\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)+\s*\|?\s*$', re.M),
]

# Cheap prefilter: if none of these characters appear, it's almost surely not Markdown.
_SENTINELS = set('#`[]()!*_>-|')

def seems_markdown(text: str, require: int = 1) -> bool:
    """
    Fast heuristic: return True if `text` likely contains Markdown formatting.
    `require` = how many distinct Markdown features must be detected (raise to 2 to lower false positives).
    """
    # O(n) character scan first (very cheap); avoids compiling/trying regexes on plain prose.
    if not any(ch in _SENTINELS for ch in text):
        return False

    hits = 0
    for rx in _MD_REGEXES:
        if rx.search(text):
            hits += 1
            if hits >= require:  # early exit on first/second feature
                return True
    return False

def check_if_markdown(text):
    """Return True if the text seems to contain Markdown formatting."""
    return is_markdown_heavy(text) or seems_markdown(text, require=3)

#endregion

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_data_path", 
        type=str, 
        default="judged_paired_2.jsonl",
        help="Path to the directory containing JSONL files with translation data."
    )
    parser.add_argument("--out_train_path", type=str, default="judged_paired_2_train.jsonl", help="Path to save the training split JSONL file.")
    parser.add_argument("--out_val_path", type=str, default="val_data.jsonl", help="Path to save the validation split JSONL file.")
    parser.add_argument("--val_count", type=int, default=1000, help="Number of samples to use for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling data before splitting.")
    return parser.parse_args()

def main(in_data_path, out_train_path, out_val_path, val_count, seed):
    # load data from the input path
    data = []
    with open(in_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    print(f"Loaded {len(data)} samples from {in_data_path}.")

    # keep only the data where 'text' field is markdown
    data = [entry for entry in data if 'text' in entry and check_if_markdown(entry['text'])]
    print(f"Total samples after filtering for markdown: {len(data)}")
    
    # shuffle data with the given seed
    random.seed(seed)
    random.shuffle(data)

    # split data into train and validation sets
    val_data = data[:val_count]
    train_data = data[val_count:]
    print("Train data size: ", len(train_data))
    print("Validation data size: ", len(val_data))
    print('---'*10)

    # save the train split in the given output path
    with open(out_train_path, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(train_data)} training samples to {out_train_path}.")

    # for the validation data only keep the following fields: 'text', 'conversation_id', 'role'
    val_data_reduced = [{k: v for k, v in entry.items() if k in ['text', 'conversation_id', 'role']} for entry in val_data]
    with open(out_val_path, 'w', encoding='utf-8') as f:
        for entry in val_data_reduced:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(val_data_reduced)} validation samples to {out_val_path}.")
    

if __name__ == "__main__":
    args = parse_args()
    in_data_path = args.in_data_path
    out_train_path = args.out_train_path
    out_val_path = args.out_val_path
    val_count = args.val_count
    seed = args.seed

    main(in_data_path, out_train_path, out_val_path, val_count, seed)


