import json
import random
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage", required=True, type=int, default=1, help="Which stage of training the files belong to (default 1)")
args = parser.parse_args()

input_dir = Path(f"dataset_s{args.stage}_sm")
images_dir = input_dir
output_dir = Path(f"dataset_s{args.stage}_sm_hf")

with open(input_dir / "metadata.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]

random.shuffle(lines)

split_ratio = 0.8
split_index = int(len(lines) * split_ratio)

train_lines = lines[:split_index]
test_lines = lines[split_index:]

print(f"Creating {len(train_lines)}-{len(test_lines)} train-test split for a total of {len(lines)} training examples")

def prepare_split(split_name, split_lines):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    with open(split_dir / "metadata.jsonl", "w") as meta_out:
        for i, entry in enumerate(tqdm(split_lines, desc=f"Copying {split_name} images")):
            if "file_name" in entry:
                old_img_path = images_dir / entry["file_name"]
                if not old_img_path.exists():
                    raise FileNotFoundError(f"Image not found: {old_img_path}")
                
                ext = old_img_path.suffix
                new_filename = f"{i:05d}{ext}"
                new_img_path = split_dir / new_filename
                
                shutil.copy(old_img_path, new_img_path)
                
                meta_out.write(json.dumps({
                    "file_name": new_filename,
                    "label": entry["label"]
                }) + "\n")
            else:
                old_img_paths = [images_dir / "images/" /f for f in entry["file_names"]]
                new_img_paths = []
                ext = old_img_paths[0].suffix
                for j, old_img_path in enumerate(old_img_paths):
                    new_filename = f"{i:05d}_t{j}{ext}"
                    new_img_path = split_dir / new_filename
                    new_img_paths.append(new_filename)
                    shutil.copy(old_img_path, new_img_path)

                meta_out.write(json.dumps({
                    "file_names": new_img_paths,
                    "origin": entry["origin"],
                    "label": entry["label"]
                }))

prepare_split("train", train_lines)
prepare_split("test", test_lines)

print(f"Finished. Output written to: {output_dir}")