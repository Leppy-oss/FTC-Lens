import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage", required=True, type=int, default="1", help="Which stage of training the files belong to (default 1)")
parser.add_argument("-f", "--hf", action="store_false", help="Whether or not to read the files from _hf or regular directory")
parser.add_argument("-d", "--disk", action="store_false", help="Whether or not to store files on disk rather than huggingface")

args = parser.parse_args()

dataset = load_dataset("imagefolder", data_dir=f"dataset_s{args.stage}_sm{"_hf" if args.hf else ""}")

if args.disk:
    dataset.save_to_disk(f"./dataset_s{args.stage}_sm_disk")
else:
    dataset.push_to_hub(f"Leppy-oss/ftc-lens-stage{args.stage}-sm")