import os
import json

def strip_folder_paths_from_file_names(jsonl_path):
    updated_lines = []

    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue  # skip empty lines
            data = json.loads(line)
            if "file_name" in data:
                data["file_name"] = os.path.basename(data["file_name"])
            updated_lines.append(json.dumps(data))

    with open(jsonl_path, "w", encoding="utf-8") as outfile:
        for updated_line in updated_lines:
            outfile.write(updated_line + "\n")


jsonl_file_path = "dataset_s2_sm/test/metadata.jsonl"
strip_folder_paths_from_file_names(jsonl_file_path)