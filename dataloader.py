from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download, login
import os
import shutil
    
def push_images_to_hub():
    ds = load_dataset(
        "imagefolder",
        data_files="processed_images/*/.png",  # Explicitly specify PNG files
        split="train"
    )
    ds.push_to_hub("tranphat1806/UI-tron")

if __name__ == "__main__":
    push_images_to_hub()