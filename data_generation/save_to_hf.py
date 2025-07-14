import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage", required=True, type=int, default="1", help="Which stage of training the files belong to (default 1)")

args = parser.parse_args()

dataset = load_dataset("imagefolder", data_dir=f"dataset_s{args.stage}_hf")

dataset.push_to_hub("Leppy-oss/FTC-Lens")