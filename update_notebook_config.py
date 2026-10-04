"""
update_notebook_config.py
Updates Section 1 (Config) in resume_screening.ipynb to use the real dataset.
"""
import json

nb_path = 'resume_screening.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Section 1 config cell is cell index 2
config_cell = nb['cells'][2]
assert 'SECTION 1' in ''.join(config_cell['source']), "Cell 2 is not the config cell!"

old_lines = config_cell['source']
new_lines = []
for line in old_lines:
    # Point to real labeled CSV
    if "DATASET_PATH = None" in line:
        new_lines.append("DATASET_PATH = 'resume_data_labeled.csv'  # Real dataset (9,544 resumes, 28 roles)\n")
    # Set explicit column names
    elif "TEXT_COLUMN  = None" in line:
        new_lines.append("TEXT_COLUMN  = 'Clean_Resume'  # Preprocessed resume text column\n")
    elif "LABEL_COLUMN = None" in line:
        new_lines.append("LABEL_COLUMN = 'job_role'     # Clean role label (28 categories)\n")
    else:
        new_lines.append(line)

config_cell['source'] = new_lines

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook config updated:")
print("  DATASET_PATH -> 'resume_data_labeled.csv'")
print("  TEXT_COLUMN  -> 'Clean_Resume'")
print("  LABEL_COLUMN -> 'job_role'")
