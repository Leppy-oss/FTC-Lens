import argparse

from huggingface_hub import upload_large_folder

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage", required=True, type=int, default="1", help="Which stage of training the files belong to (default 1)")

args = parser.parse_args()

upload_large_folder(
    folder_path=f"dataset_s{args.stage}",
    repo_id="Leppy-oss/FTC-Lens",
    repo_type="dataset",
)