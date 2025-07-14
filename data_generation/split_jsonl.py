import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage", required=True, type=int, default="1", help="Which stage of training the files belong to (default 1)")

args = parser.parse_args()

with open(f"dataset_s{args.stage}/metadata.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]

random.shuffle(lines)

split_ratio = 0.8
split_index = int(len(lines) * split_ratio)

train_lines = lines[:split_index]
test_lines = lines[split_index:]

print(f"Created {len(train_lines)}-{len(test_lines)} train-test split for a total of {len(lines)} training samples")

with open(f"dataset_s{args.stage}/train.jsonl", "w") as f_train:
    for entry in train_lines:
        f_train.write(json.dumps(entry) + "\n")

with open(f"dataset_s{args.stage}/test.jsonl", "w") as f_test:
    for entry in test_lines:
        f_test.write(json.dumps(entry) + "\n")
