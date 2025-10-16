import os, json

in_data_path = "/workspace/translation_optimization/get_translations/nemotron_translations"
data = []
for file in sorted(os.listdir(in_data_path)):
    if file.endswith(".jsonl"):
        file_data = []
        with open(os.path.join(in_data_path, file), 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    file_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file}: {e}")
                    continue
        data = data + file_data
        
        # print(f"There are {len(file_data)} entries in {file}, out of which {len([d for d in file_data if d['role'] == 'assistant'])} are from the assistant and {len([d for d in file_data if d['role'] == 'user'])} from the user.")

assistant_data = [d for d in data if d['role'] == 'assistant']
user_data = [d for d in data if d['role'] == 'user']


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

#region detecting markdown with regexes
import re

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
#endregion

markdown_assistant_data = [d for d in assistant_data if is_markdown_heavy(d['text']) or seems_markdown(d['text'], require=3)]
print(f"There are {len(markdown_assistant_data)} heavy entries out of {len(assistant_data)} assistant entries ({len(markdown_assistant_data)/len(assistant_data)*100:.2f}%).")

print("Fields in the data: ", markdown_assistant_data[0].keys())

# # save the markdown assistant data to a jsonl file
# out_path = "filtered/markdown_assistant_data.jsonl"
# with open(out_path, 'w', encoding='utf-8') as f:
#     for entry in markdown_assistant_data:
#         f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# print(f"Saved markdown assistant data to {out_path}.")