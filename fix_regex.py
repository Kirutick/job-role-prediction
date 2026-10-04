import json
import re

path = 'resume_screening.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if r"r'\b(" in line:
                line = line.replace(r"r'\b(", r"r'(?<!\w)(")
            if r")\b'" in line:
                line = line.replace(r")\b'", r")(?!\w)'")
            new_source.append(line)
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
